// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "render/filament/core/texture.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <utility>

#include <filament/Engine.h>
#include <filament/Texture.h>
#include <image/Ktx1Bundle.h>
#include <ktxreader/Ktx1Reader.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/releaser.h"

namespace mujoco {

static std::atomic<uint64_t> g_next_texture_id{1};

static constexpr int kNumFacesPerCube = 6;

static bool IsCompressed(const mjrfTextureConfig& config) {
  return config.format == mjPIXEL_FORMAT_KTX;
}

static bool IsCubeMap(const mjrfTextureConfig& config) {
  return config.sampler_type == mjTEXTURE_CUBE ||
         config.sampler_type == mjTEXTURE_SKYBOX;
}

static int GetFaceHeight(const mjrfTextureConfig& config) {
  int face_height = config.height;
  if (config.width != config.height) {
    if (config.width * kNumFacesPerCube != config.height) {
      mju_error("Cube maps must contain 6 square images.");
    }
    face_height = config.height / kNumFacesPerCube;
  }
  if (config.width != face_height) {
    mju_error("Cube map faces must be square.");
  }
  return face_height;
}

static int GetNumChannels(const mjrfTextureConfig& config) {
  switch (config.format) {
    case mjPIXEL_FORMAT_R8:
      return 1;
    case mjPIXEL_FORMAT_RGB8:
      return 3;
    case mjPIXEL_FORMAT_RGBA8:
      return 4;
    default:
      mju_error("Unsupported format: %d", (int)config.format);
      return 0;
  }
}

static filament::Texture::Format GetTextureFormat(
    const mjrfTextureConfig& config) {
  switch (config.format) {
    case mjPIXEL_FORMAT_R8:
      return filament::Texture::Format::R;
    case mjPIXEL_FORMAT_RGB8:
      return filament::Texture::Format::RGB;
    case mjPIXEL_FORMAT_RGBA8:
      return filament::Texture::Format::RGBA;
    default:
      mju_error("Unsupported format: %d", (int)config.format);
      return filament::Texture::Format::UNUSED;
  }
}

static filament::Texture::InternalFormat GetTextureInternalFormat(
    const mjrfTextureConfig& config) {
  if (config.color_space == mjCOLORSPACE_SRGB) {
    switch (config.format) {
      case mjPIXEL_FORMAT_RGB8:
        return filament::Texture::InternalFormat::SRGB8;
      case mjPIXEL_FORMAT_RGBA8:
        return filament::Texture::InternalFormat::SRGB8_A8;
      default:
        mju_error("Unsupported format: %d", (int)config.format);
        return filament::Texture::InternalFormat::UNUSED;
    }
  } else {
    switch (config.format) {
      case mjPIXEL_FORMAT_R8:
        return filament::Texture::InternalFormat::R8;
      case mjPIXEL_FORMAT_RGB8:
        return filament::Texture::InternalFormat::RGB8;
      case mjPIXEL_FORMAT_RGBA8:
        return filament::Texture::InternalFormat::RGBA8;
      case mjPIXEL_FORMAT_R32F:
        return filament::Texture::InternalFormat::R32F;
      case mjPIXEL_FORMAT_DEPTH32F:
        return filament::Texture::InternalFormat::DEPTH32F;
      default:
        mju_error("Unsupported format: %d", (int)config.format);
        return filament::Texture::InternalFormat::UNUSED;
    }
  }
}

Texture::Texture(filament::Engine* engine, const mjrfTextureConfig& config,
                 InternalFlags flags)
    : id_(g_next_texture_id++), engine_(engine), config_(config) {
  if (IsCompressed(config_)) {
    // We defer creation of compressed textures until Upload() is called. In
    // the meantime, we don't really know anything about the texture (e.g.
    // width, height, etc.).
    return;
  }

  const filament::Texture::InternalFormat internal_format =
      GetTextureInternalFormat(config_);
  filament::Texture::Builder builder;
  builder.width(config_.width);
  builder.height(config_.height);
  builder.format(internal_format);

  if (IsCubeMap(config_)) {
    if (config_.format != mjPIXEL_FORMAT_RGB8) {
      mju_error("Only support RGB cubemaps.");
      return;
    }
    builder.height(GetFaceHeight(config_));
    builder.sampler(filament::Texture::Sampler::SAMPLER_CUBEMAP);
  } else {
    builder.sampler(filament::Texture::Sampler::SAMPLER_2D);
  }

  filament::Texture::Usage usage = filament::Texture::Usage::DEFAULT;
  if (flags.color_attachment) {
    usage |= filament::Texture::Usage::COLOR_ATTACHMENT;
    usage |= filament::Texture::Usage::BLIT_SRC;
  } else if (flags.depth_attachment) {
    usage |= filament::Texture::Usage::DEPTH_ATTACHMENT;
    usage |= filament::Texture::Usage::BLIT_SRC;
  } else if (filament::Texture::isTextureFormatMipmappable(*engine_,
                                                           internal_format)) {
    usage |= filament::Texture::Usage::GEN_MIPMAPPABLE;
    builder.levels(0xff);
  }
  builder.usage(usage);

  texture_ = builder.build(*engine_);
}

Texture::~Texture() {
  if (texture_) {
    engine_->destroy(texture_);
  }
}

void Texture::Upload(const mjrfTextureData& data) {
  Releaser releaser;
  releaser.Add(data.release, data.user_data);

  if (data.bytes == nullptr || data.num_bytes == 0) {
    return;
  }

  if (config_.format == mjPIXEL_FORMAT_KTX) {
    image::Ktx1Bundle* bundle = new image::Ktx1Bundle(
        reinterpret_cast<const uint8_t*>(data.bytes), data.num_bytes);
    has_spherical_harmonics_ = true;
    bundle->getSphericalHarmonics(spherical_harmonics_);
    auto format = ktxreader::Ktx1Reader::toTextureFormat(bundle->getInfo());
    const bool is_srgb = ktxreader::Ktx1Reader::isSrgbTextureFormat(format);
    texture_ = ktxreader::Ktx1Reader::createTexture(engine_, bundle, is_srgb);
    config_.width = texture_->getWidth();
    config_.height = texture_->getHeight();
    return;
  }

  const int num_channels = GetNumChannels(config_);
  const filament::Texture::Type type = filament::Texture::Type::UBYTE;
  const filament::Texture::Format format = GetTextureFormat(config_);

  if (!IsCubeMap(config_)) {
    if (config_.width * config_.height * num_channels != data.num_bytes) {
      mju_error("Texture size does not match data size.");
    }

    filament::Texture::PixelBufferDescriptor desc(
        data.bytes, data.num_bytes, format, type, &Releaser::Release,
        releaser.Acquire());
    texture_->setImage(*engine_, 0, std::move(desc));
  } else {
    const int face_size = config_.width * GetFaceHeight(config_) * num_channels;
    const int num_bytes = face_size * kNumFacesPerCube;

    if (config_.width == config_.height) {
      // Create a new image buffer into which we can make 6 copies of the input
      // image to create a cubemap. Because this will be the actual buffer
      // passed to Filament, we use a new Releaser and allow the original to go
      // out of scope.
      Releaser copy_releaser;
      uint8_t* copy = copy_releaser.Alloc<uint8_t>(num_bytes);
      for (int i = 0; i < kNumFacesPerCube; ++i) {
        std::memcpy(copy + (i * face_size), data.bytes, face_size);
      }
      filament::Texture::PixelBufferDescriptor desc(copy, num_bytes, format,
                                                    type, &Releaser::Release,
                                                    copy_releaser.Acquire());
      texture_->setImage(*engine_, /*level=*/0, /*xoffset=*/0, /*yoffset=*/0,
                         /*zoffset=*/0, config_.width, GetFaceHeight(config_),
                         /*depth=*/6, std::move(desc));
    } else {
      if (num_bytes != data.num_bytes) {
        mju_error("Texture size does not match data size.");
      }
      filament::Texture::PixelBufferDescriptor desc(
          data.bytes, data.num_bytes, format, type, &Releaser::Release,
          releaser.Acquire());
      texture_->setImage(*engine_, /*level=*/0, /*xoffset=*/0, /*yoffset=*/0,
                         /*zoffset=*/0, config_.width, GetFaceHeight(config_),
                         /*depth=*/6, std::move(desc));
    }
  }

  if (texture_->getLevels() > 1) {
    texture_->generateMipmaps(*engine_);
  }
}

mjrSamplerType Texture::GetSamplerType() const {
  return static_cast<mjrSamplerType>(config_.sampler_type);
}
}  // namespace mujoco
