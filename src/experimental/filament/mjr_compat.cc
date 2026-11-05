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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "experimental/filament/compat/scene_bridge.h"
#include "render/filament/support/mjrf_ptr.h"

// This library implements the entirety of mujoco's mjr API. You can link this
// library with your application (instead of the "classic" mujoco renderer) to
// use the same APIs but with Filament rendering instead. Note that some
// functionality (e.g. ux-related functions like mjr_text, mjr_label, etc.)
// will call mju_error if called.
//
// You should consider using filament's mjrf API directly as it will provide you
// with access to more features and optimizations.

namespace mujoco {
namespace {

class CompatContext {
 public:
  CompatContext(const mjrfContextConfig* config, const mjModel* model);

  void Render(const mjrRect& viewport, const mjvScene* scene);

  void ReadPixels(mjrRect viewport, unsigned char* rgb, float* depth,
                  int depth_map);

  // The headlight is a render request option; fill it in from the bridge.
  void SetHeadlight(mjrfRenderRequest* req) const {
    req->headlight = scene_bridge_->IsHeadlightEnabled();
    const float* color = scene_bridge_->GetHeadlightColor();
    req->headlight_color[0] = color[0];
    req->headlight_color[1] = color[1];
    req->headlight_color[2] = color[2];
    req->headlight_intensity = scene_bridge_->GetHeadlightIntensity();
  }

  void SetFrameBuffer(int framebuffer) {
    framebuffer_ = (mjtFramebuffer)framebuffer;
  }

  void UploadMesh(const mjModel* model, int id) {
    scene_bridge_->UploadMesh(model, id);
  }

  void UploadTexture(const mjModel* model, int id) {
    scene_bridge_->UploadTexture(model, id);
  }

  void UploadHeightField(const mjModel* model, int id) {
    scene_bridge_->UploadHeightField(model, id);
  }

  mjrfContext* Context() const { return context_.get(); }

 private:
  mjrDrawMode draw_mode_ = mjDRAW_MODE_DEFAULT;
  bool skybox_ = true;
  bool haze_ = true;
  UniquePtr<mjrfContext> context_{nullptr, nullptr};
  UniquePtr<mjrfScene> scene_{nullptr, nullptr};
  std::unique_ptr<SceneBridge> scene_bridge_;
  mjtFramebuffer framebuffer_ = mjFB_WINDOW;
  UniquePtr<mjrfRenderTarget> color_target_{nullptr, nullptr};
  UniquePtr<mjrfRenderTarget> depth_target_{nullptr, nullptr};
};

CompatContext::CompatContext(const mjrfContextConfig* config,
                             const mjModel* model) {
  context_ = CreateContext(*config);
  scene_ = CreateScene(context_.get(), {});
  scene_bridge_ =
      std::make_unique<SceneBridge>(context_.get(), scene_.get(), model);
}

void CompatContext::Render(const mjrRect& viewport, const mjvScene* scene) {
  scene_bridge_->Update(viewport, scene);

  if (scene->flags[mjRND_SEGMENT]) {
    if (scene->flags[mjRND_IDCOLOR]) {
      draw_mode_ = mjDRAW_MODE_SEGMENTATION_BY_ID;
    } else {
      draw_mode_ = mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    }
  } else if (scene->flags[mjRND_DEPTH]) {
    draw_mode_ = mjDRAW_MODE_DEPTH;
  } else if (scene->flags[mjRND_WIREFRAME]) {
    draw_mode_ = mjDRAW_MODE_WIREFRAME;
  } else {
    draw_mode_ = mjDRAW_MODE_DEFAULT;
  }
  skybox_ = draw_mode_ == mjDRAW_MODE_DEFAULT && scene->flags[mjRND_SKYBOX];
  haze_ = draw_mode_ == mjDRAW_MODE_DEFAULT && scene->flags[mjRND_HAZE];

  if (framebuffer_ == mjFB_WINDOW) {
    mjrfRenderRequest req;
    mjrf_defaultRenderRequest(&req);
    req.scene = scene_.get();
    req.draw_mode = draw_mode_;
    req.skybox = skybox_;
    req.haze = haze_;
    req.camera = scene_bridge_->GetCamera();
    req.viewport = viewport;
    SetHeadlight(&req);
    mjrf_render(context_.get(), &req, 1, nullptr, 0);
  }
}

void CompatContext::ReadPixels(mjrRect viewport, unsigned char* rgb,
                               float* depth, int depth_map) {
  if (framebuffer_ == mjFB_WINDOW) {
    mju_error("ReadPixels is only supported for offscreen rendering.");
  }

  const int width = viewport.width;
  const int height = viewport.height;

  if (rgb) {
    if (!color_target_) {
      mjrfRenderTargetConfig config;
      mjrf_defaultRenderTargetConfig(&config);
      config.color_format = mjPIXEL_FORMAT_RGB8;
      config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
      color_target_ = CreateRenderTarget(context_.get(), config);
    }
    mjrf_resizeRenderTarget(color_target_.get(), width, height);

    mjrfRenderRequest req;
    mjrf_defaultRenderRequest(&req);
    req.scene = scene_.get();
    req.draw_mode = draw_mode_;
    req.skybox = skybox_;
    req.haze = haze_;
    req.camera = scene_bridge_->GetCamera();
    req.viewport = viewport;
    req.target = color_target_.get();
    SetHeadlight(&req);

    mjrfReadRequest read_req;
    mjrf_defaultReadRequest(&read_req);
    read_req.target = color_target_.get();
    read_req.output = rgb;
    read_req.num_bytes = viewport.width * viewport.height * 3;

    const mjrfFrameHandle frame = mjrf_render(context_.get(), &req, 1, &read_req, 1);
    mjrf_waitForFrame(context_.get(), frame);
  }

  if (depth) {
    if (!depth_target_) {
      mjrfRenderTargetConfig config;
      mjrf_defaultRenderTargetConfig(&config);
      config.color_format = mjPIXEL_FORMAT_R32F;
      config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
      depth_target_ = CreateRenderTarget(context_.get(), config);
    }
    mjrf_resizeRenderTarget(depth_target_.get(), width, height);

    mjrfRenderRequest req;
    mjrf_defaultRenderRequest(&req);
    req.scene = scene_.get();
    req.draw_mode = mjDRAW_MODE_DEPTH;
    req.skybox = false;
    req.haze = false;
    req.camera = scene_bridge_->GetCamera();
    req.viewport = viewport;
    req.target = depth_target_.get();

    mjrfReadRequest read_req;
    mjrf_defaultReadRequest(&read_req);
    read_req.target = depth_target_.get();
    read_req.output = reinterpret_cast<uint8_t*>(depth);
    read_req.num_bytes = viewport.width * viewport.height * sizeof(float);

    const mjrfFrameHandle frame = mjrf_render(context_.get(), &req, 1, &read_req, 1);
    mjrf_waitForFrame(context_.get(), frame);
    if (depth_map == mjDEPTH_ZERONEAR) {
      const size_t npixel =
          static_cast<size_t>(width) * static_cast<size_t>(height);
      for (size_t i = 0; i < npixel; ++i) {
        depth[i] = 1.0f - depth[i];
      }
    }
  }
}

}  // namespace
}  // namespace mujoco

static thread_local mujoco::CompatContext* g_context = nullptr;

static mujoco::CompatContext* GetCheckedContext() {
  if (g_context == nullptr) {
    mju_error("Missing context; did you call mjr_makeContext?");
  }
  return g_context;
}

extern "C" {

// mjr functions that are supported by the filament renderer.

void mjr_defaultContext(mjrContext* con) {
  if (con) {
    memset(con, 0, sizeof(mjrContext));
  }
}

void mjr_defaultRendererInfo(mjrRendererInfo* info) {
  memset(info, 0, sizeof(mjrRendererInfo));
  info->renderer = "filament";
  info->backend = "";
}

void mjr_getRendererInfo(mjrRendererInfo* info) {
  mjr_defaultRendererInfo(info);
  if (g_context) {
    mjrf_getRendererInfo(g_context->Context(), info);
  }
}

void mjr_makeFilamentContext(const mjModel* m, const mjrfContextConfig* cfg,
                             mjrContext* con) {
  if (g_context != nullptr) {
    mju_error("Context already exists!");
  }
  mjr_defaultContext(con);
  g_context = new mujoco::CompatContext(cfg, m);
}

void mjr_makeContext(const mjModel* m, mjrContext* con, int fontscale) {
  mjr_freeContext(con);

  mjrfContextConfig cfg;
  mjrf_defaultContextConfig(&cfg);
  mjr_makeFilamentContext(m, &cfg, con);
}

void mjr_freeContext(mjrContext* con) {
  // mjr_freeContext may be called multiple times.
  if (g_context) {
    delete g_context;
    g_context = nullptr;
  }
  mjr_defaultContext(con);
}

void mjr_render(mjrRect viewport, mjvScene* scn, const mjrContext* con) {
  GetCheckedContext()->Render(viewport, scn);
}
void mjr_uploadMesh(const mjModel* m, const mjrContext* con, int meshid) {
  GetCheckedContext()->UploadMesh(m, meshid);
}
void mjr_uploadTexture(const mjModel* m, const mjrContext* con, int texid) {
  GetCheckedContext()->UploadTexture(m, texid);
}
void mjr_uploadHField(const mjModel* m, const mjrContext* con, int hfieldid) {
  GetCheckedContext()->UploadHeightField(m, hfieldid);
}
void mjr_setBuffer(int framebuffer, mjrContext* con) {
  GetCheckedContext()->SetFrameBuffer(framebuffer);
}
void mjr_readPixels(unsigned char* rgb, float* depth, mjrRect viewport,
                    const mjrContext* con) {
  const int depth_map = con ? con->readDepthMap : mjDEPTH_ZERONEAR;
  GetCheckedContext()->ReadPixels(viewport, rgb, depth, depth_map);
}

// mjr functions that are NOT supported by the filament renderer.

void mjr_setAux(int index, const mjrContext* con) {
  mju_error("mjr_setAux not implemented.");
}
void mjr_restoreBuffer(const mjrContext* con) {
  mju_error("mjr_restoreBuffer not implemented.");
}
void mjr_rectangle(mjrRect viewport, float r, float g, float b, float a) {
  mju_error("mjr_rectangle not implemented.");
}
void mjr_blitAux(int index, mjrRect src, int left, int bottom,
                 const mjrContext* con) {
  mju_error("mjr_blitAux not implemented.");
}
void mjr_changeFont(int fontscale, mjrContext* con) {
  mju_error("mjr_changeFont not implemented.");
}
void mjr_addAux(int index, int width, int height, int samples,
                mjrContext* con) {
  mju_error("mjr_addAux not implemented.");
}
void mjr_resizeOffscreen(int width, int height, mjrContext* con) {
  mju_error("mjr_resizeOffscreen not implemented.");
}
void mjr_drawPixels(const unsigned char* rgb, const float* depth,
                    mjrRect viewport, const mjrContext* con) {
  mju_error("mjr_drawPixels not implemented.");
}
void mjr_blitBuffer(mjrRect src, mjrRect dst, int flg_color, int flg_depth,
                    const mjrContext* con) {
  mju_error("mjr_blitBuffer not implemented.");
}
void mjr_text(int font, const char* txt, const mjrContext* con, float x,
              float y, float r, float g, float b) {
  mju_error("mjr_text not implemented.");
}
void mjr_overlay(int font, int gridpos, mjrRect viewport, const char* overlay,
                 const char* overlay2, const mjrContext* con) {
  mju_error("mjr_overlay not implemented.");
}
void mjr_label(mjrRect viewport, int font, const char* txt, float r, float g,
               float b, float a, float rt, float gt, float bt,
               const mjrContext* con) {
  mju_error("mjr_label not implemented.");
}
void mjr_figure(mjrRect viewport, mjvFigure* fig, const mjrContext* con) {
  mju_error("mjr_figure not implemented.");
}
void mjr_finish() { mju_error("mjr_finish not implemented."); }
int mjr_getError() {
  mju_error("mjr_getError not implemented.");
  return 0;
}
mjrRect mjr_maxViewport(const mjrContext* con) {
  mju_error("mjr_maxViewport not implemented.");
  return mjrRect{};
}
int mjr_findRect(int x, int y, int nrect, const mjrRect* rect) {
  mju_error("mjr_findRect not implemented.");
  return 0;
}

}  // extern "C"
