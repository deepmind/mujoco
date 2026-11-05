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

#include "studio/hal/filament_renderer.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <span>

#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/imgui_editor.h"
#include "render/filament/support/imgui_bridge.h"
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_renderables.h"
#include "render/filament/support/model_resources.h"
#include "studio/hal/graphics_mode.h"
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/plugin.h"

namespace mujoco::studio {

static std::function<void()> g_update_gui_callback = nullptr;
static void PluginUpdate(GuiPlugin* plugin) {
  if (g_update_gui_callback) {
    g_update_gui_callback();
  }
}

FilamentRenderer::FilamentRenderer(void* native_window, GraphicsMode gfx) {
  g_update_gui_callback = [this]() {
    if (main_scene_) {
      DrawSceneEditor(main_scene_.get());
    }
  };

  for (int i = 0; i < mjNRNDFLAG; i++) {
    render_flags_[i] = (mjRNDSTRING[i][1][0] == '1');
  }

  mjrfContextConfig cfg;
  mjrf_defaultContextConfig(&cfg);
  cfg.native_window = native_window;
  cfg.software_rendering = IsSoftware(gfx);
  cfg.graphics_api = IsOpenGl(gfx) || IsWebGl(gfx) ? mjGRAPHICS_API_OPENGL : mjGRAPHICS_API_VULKAN;
  filament_context_ = CreateContext(cfg);

  ux_scene_ = CreateScene(filament_context_.get(), {});
  imgui_bridge_ = std::make_unique<ImguiBridge>(filament_context_.get(), ux_scene_.get());

  mjrfRenderTargetConfig config;
  mjrf_defaultRenderTargetConfig(&config);
  config.color_format = mjPIXEL_FORMAT_RGB8;
  config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  render_target_ = CreateRenderTarget(filament_context_.get(), config);
}

FilamentRenderer::~FilamentRenderer() { g_update_gui_callback = nullptr; }

void FilamentRenderer::Init(const mjModel* model) {
  model_decorations_.reset();
  model_renderables_.reset();
  model_lights_.reset();
  model_resources_.reset();
  main_scene_.reset();

  if (model) {
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const int id = mj_name2id(model, mjOBJ_NUMERIC, "filament.clear_color");
    if (id >= 0 && model->numeric_size[id] == 4) {
      const mjtNum* ptr = model->numeric_data + model->numeric_adr[id];
      for (int i = 0; i < 4; ++i) {
        clear_color[i] = static_cast<float>(ptr[i]);
      }
    }
    mjrf_setClearColor(filament_context_.get(), &clear_color[0]);

    main_scene_ = CreateScene(filament_context_.get(), {});
    mjrf_configureSceneFromModel(main_scene_.get(), model);
    offsamples_ = model->vis.quality.offsamples;
    model_resources_ = std::make_unique<ModelResources>(model, filament_context_.get());
    model_lights_ = std::make_unique<ModelLights>(main_scene_.get(), model_resources_.get());
    model_renderables_ =
        std::make_unique<ModelRenderables>(main_scene_.get(), model_resources_.get());
    model_decorations_ =
        std::make_unique<ModelDecorations>(filament_context_.get(), main_scene_.get(), model);
  }
}

void FilamentRenderer::Sync(const mjModel* model, mjData* data, const mjvPerturb* perturb,
                            mjvCamera* camera, const mjvOption* vis_option, int width, int height,
                            std::span<mjvGeom> extra_geoms) {
  const mjrRect viewport = {0, 0, width, height};

  mjvCamera default_cam;
  if (camera == nullptr) {
    if (model) {
      mjv_defaultFreeCamera(model, &default_cam);
    } else {
      mjv_defaultCamera(&default_cam);
    }
    camera = &default_cam;
  }

  mjvOption default_opt;
  if (vis_option == nullptr) {
    mjv_defaultOption(&default_opt);
    vis_option = &default_opt;
  }

  mjvPerturb default_perturb;
  if (perturb == nullptr) {
    mjv_defaultPerturb(&default_perturb);
    perturb = &default_perturb;
  }

  const mjrCamera gl_camera = mjv_camera2GLCamera(model, data, camera);

  model_lights_->Update(data);

  if (model->vis.quality.offsamples != offsamples_) {
    offsamples_ = model->vis.quality.offsamples;
    mjrf_setSceneMsaa(main_scene_.get(), offsamples_);
  }

  if (vis_option) {
    model_renderables_->SetOptions(*vis_option);
  }
  if (perturb->select > 0) {
    model_renderables_->MarkAsSelected(mjOBJ_BODY, perturb->select);
  } else if (perturb->flexselect >= 0) {
    model_renderables_->MarkAsSelected(mjOBJ_FLEX, perturb->flexselect);
  } else if (perturb->skinselect >= 0) {
    model_renderables_->MarkAsSelected(mjOBJ_SKIN, perturb->skinselect);
  } else {
    model_renderables_->MarkAsSelected(mjOBJ_UNKNOWN, -1);
  }
  model_renderables_->Update(data);

  model_decorations_->Update(data, vis_option, perturb, camera, viewport, DrawTextAt, extra_geoms);

  BuildMainRenderRequest(&render_requests_[0], model->vis, *vis_option, viewport, gl_camera);
  BuildUxRenderRequest(&render_requests_[1], viewport);
}

void FilamentRenderer::Submit(int width, int height, std::span<std::byte> pixels) {
  const mjrRect viewport = {0, 0, width, height};

  imgui_bridge_->Update();

  mjrfFrameHandle frame = 0;
  if (pixels.empty()) {
    frame = mjrf_render(filament_context_.get(), &render_requests_[0], 2, nullptr, 0);
  } else {
    if (pixels.size() != width * height * 3) {
      mju_error("Offscreen mode requires a pixel buffer of size %d.", width * height * 3);
    }

    mjrf_resizeRenderTarget(render_target_.get(), width, height);
    render_requests_[0].target = render_target_.get();
    render_requests_[1].target = render_target_.get();

    mjrfReadRequest read_request;
    mjrf_defaultReadRequest(&read_request);
    read_request.target = render_target_.get();
    read_request.output = pixels.data();
    read_request.num_bytes = viewport.width * viewport.height * 3;

    frame = mjrf_render(filament_context_.get(), &render_requests_[0], 2, &read_request, 1);
  }

  mjrf_waitForFrame(filament_context_.get(), frame);

  mjrfFrameStats stats;
  mjrf_defaultFrameStats(&stats);
  mjrf_getFrameStats(filament_context_.get(), frame, &stats);
  gpu_frame_ms_ = stats.frame_rate > 0 ? 1000.0 / stats.frame_rate : 0;
}

void FilamentRenderer::RenderToTexture(const mjModel* model, mjData* data, mjvCamera* camera,
                                       int width, int height, std::byte* output) {
  if (!output) {
    return;
  }

  mjrf_resizeRenderTarget(render_target_.get(), width, height);

  mjvOption vis_option;
  mjv_defaultOption(&vis_option);

  mjrfRenderRequest request;
  BuildMainRenderRequest(&request, model->vis, vis_option, {0, 0, width, height},
                         mjv_camera2GLCamera(model, data, camera));
  request.target = render_target_.get();

  mjrfReadRequest read_request;
  mjrf_defaultReadRequest(&read_request);
  read_request.target = render_target_.get();
  read_request.output = output;
  read_request.num_bytes = width * height * 3;

  const mjrfFrameHandle frame = mjrf_render(filament_context_.get(), &request, 1, &read_request, 1);
  mjrf_waitForFrame(filament_context_.get(), frame);
}

int FilamentRenderer::UploadImage(int tex_id, std::span<const std::byte> pixels, int width,
                                  int height) {
  return imgui_bridge_->UploadImage(tex_id, pixels, width, height);
}

double FilamentRenderer::GetGpuFrameMs() { return gpu_frame_ms_; }

void FilamentRenderer::BuildMainRenderRequest(mjrfRenderRequest* request, const mjVisual& vis,
                                              const mjvOption& vis_option, const mjrRect& viewport,
                                              const mjrCamera& camera) {
  mjrDrawMode draw_mode = mjDRAW_MODE_DEFAULT;
  if (render_flags_[mjRND_SEGMENT]) {
    if (render_flags_[mjRND_IDCOLOR]) {
      draw_mode = mjDRAW_MODE_SEGMENTATION_BY_ID;
    } else {
      draw_mode = mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    }
  } else if (render_flags_[mjRND_DEPTH]) {
    draw_mode = mjDRAW_MODE_DEPTH;
  } else if (render_flags_[mjRND_WIREFRAME]) {
    draw_mode = mjDRAW_MODE_WIREFRAME;
  } else if (vis_option.flags[mjVIS_ISLAND]) {
    draw_mode = mjDRAW_MODE_ISLANDS;
  }

  mjrf_defaultRenderRequest(request);
  request->scene = main_scene_.get();
  request->draw_mode = draw_mode;
  request->camera = camera;
  request->viewport = viewport;
  request->shadows = render_flags_[mjRND_SHADOW];
  request->reflections = render_flags_[mjRND_REFLECTION];
  request->post_processing = render_flags_[mjRND_POST_PROCESSING];
  request->headlight = vis.headlight.active;
  request->headlight_color[0] = vis.headlight.diffuse[0];
  request->headlight_color[1] = vis.headlight.diffuse[1];
  request->headlight_color[2] = vis.headlight.diffuse[2];
  request->headlight_intensity = model_lights_->GetHeadlightIntensity();
}

void FilamentRenderer::BuildUxRenderRequest(mjrfRenderRequest* request, const mjrRect& viewport) {
  mjrf_defaultRenderRequest(request);
  request->scene = ux_scene_.get();
  request->draw_mode = mjDRAW_MODE_DEFAULT;
  request->camera.orthographic = true;
  request->camera.pos[0] = 0.0f;
  request->camera.pos[1] = 0.0f;
  request->camera.pos[2] = 1.0f;
  request->camera.forward[0] = 0.0f;
  request->camera.forward[1] = 0.0f;
  request->camera.forward[2] = -1.0f;
  request->camera.up[0] = 0.0f;
  request->camera.up[1] = 1.0f;
  request->camera.up[2] = 0.0f;
  request->camera.frustum_top = 0.0f;
  request->camera.frustum_near = 0.0f;
  request->camera.frustum_far = 1.0f;
  request->camera.frustum_center = viewport.width / 2.0f;
  request->camera.frustum_width = viewport.width / 2.0f;
  request->camera.frustum_bottom = viewport.height;
  request->viewport = viewport;
  request->shadows = false;
  request->reflections = false;
  request->post_processing = false;
}
}  // namespace mujoco::studio

mjPLUGIN_LIB_INIT(renderer) {
  mujoco::studio::GuiPlugin plugin;
  plugin.name = "Filament";
  plugin.update = [](mujoco::studio::GuiPlugin* self) { mujoco::studio::PluginUpdate(self); };
  mujoco::studio::RegisterPlugin(plugin);
}
