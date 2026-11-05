// Copyright 2026 DeepMind Technologies Limited
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

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(__APPLE__)
  #include <mach-o/dyld.h>
  #include <cstdint>
#elif defined(_WIN32)
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#include "lodepng.h"
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/model_decorations.h"
#include "render/filament/support/model_lights.h"
#include "render/filament/support/model_resources.h"
#include "render/filament/support/model_renderables.h"

// bitmap font definitions for CPU text overlay with Filament backend
#include "render/classic/font/normal150.inc"
#include "render/classic/font/back150.inc"

using std::optional;
using std::string;
using std::string_view;
using std::vector;

using ::mujoco::ModelDecorations;
using ::mujoco::ModelLights;
using ::mujoco::ModelRenderables;
using ::mujoco::ModelResources;


//-------------------------------- help & options --------------------------------------------------

// help message
static const char* help_msg =
    "\n"
    "Usage:  render model [output] [options]\n"
    "\n"
    "  argument/option       default       semantic\n"
    "  ---------------       -------       --------\n"
    "  model                               path to model (required, positional)\n"
    "  output                <model>.png   path to output PNG image file (positional)\n"
    "  --backend=B           filament      rendering backend (filament, classic)\n"
    "  --width=N             640           image width in pixels\n"
    "  --height=N            480           image height in pixels\n"
    "  --camera=C            (free)        camera name or integer index\n"
    "  --lookat=X,Y,Z        (center)      free camera lookat position\n"
    "  --distance=D          (fit)         free camera distance\n"
    "  --azimuth=A           90            free camera azimuth in degrees\n"
    "  --elevation=E         -45           free camera elevation in degrees\n"
    "  --key=K               (none)        keyframe name or integer index\n"
    "  --steps=N             0             simulation steps to advance before rendering\n"
    "  --geomgroup=G         111000        6-character string enabling/disabling geom groups 0-5\n"
    "  --sitegroup=S         111000        6-character string enabling/disabling site groups 0-5\n"
    "  --label=L             (none)        type of label (e.g. --label=geom, --label=body)\n"
    "  --frame=F             (none)        type of frame (e.g. --frame=geom, --frame=body)\n"
    "  --<visflag>=0|1       (default)     toggle visualization flag (e.g. --joint=1)\n"
    "  --<rndflag>=0|1       (default)     toggle rendering flag (e.g. --shadow=0, --wireframe=1)\n"
    "  --help (or no arguments)            print this help message\n";


// command-line options
struct Options {
  string model_path;
  string output_path;
  string backend = "filament";
  int    width   = 640;
  int    height  = 480;
  string camera;
  string key;
  string geomgroup;
  string sitegroup;
  int    steps = 0;
  int    label = -1;
  int    frame = -1;

  // free camera controls
  optional<std::array<mjtNum, 3>> lookat;
  optional<mjtNum>                distance;
  optional<mjtNum>                azimuth;
  optional<mjtNum>                elevation;

  // visual and rendering flags (-1 = default)
  int vis_flags[mjNVISFLAG];
  int rnd_flags[mjNRNDFLAG];

  Options() {
    std::fill_n(vis_flags, mjNVISFLAG, -1);
    std::fill_n(rnd_flags, mjNRNDFLAG, -1);
  }
};


//-------------------------------- command-line parsing --------------------------------------------

// helper: convert flag name to lower-case string without spaces or punctuation
static string NormalizeName(string_view name) {
  string result;
  result.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      result.push_back(std::tolower(static_cast<unsigned char>(c)));
    }
  }
  return result;
}


// helper: parse boolean value (1/0, true/false, on/off)
static bool ParseBool(string_view val, int* out) {
  if (val == "1" || val == "true" || val == "on") {
    *out = 1;
    return true;
  }
  if (val == "0" || val == "false" || val == "off") {
    *out = 0;
    return true;
  }
  return false;
}


// helper: parse integer value
static bool ParseInt(string_view val, int* out, int min_val = 1, int max_val = 100000) {
  int v          = 0;
  auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), v);
  if (ec != std::errc() || ptr != val.data() + val.size() || v < min_val || v > max_val) {
    return false;
  }
  *out = v;
  return true;
}


// helper: parse mjtNum value
static bool ParseNum(string_view val, mjtNum* out, mjtNum min_val = -1e9, mjtNum max_val = 1e9) {
  char*  end = nullptr;
  string str(val);
  double v = std::strtod(str.c_str(), &end);
  if (end == str.c_str() || *end != '\0' || v < min_val || v > max_val) { return false; }
  *out = static_cast<mjtNum>(v);
  return true;
}


// helper: parse mjtNum value into optional
static bool ParseNum(string_view       val,
                     optional<mjtNum>& out,
                     mjtNum            min_val = -1e9,
                     mjtNum            max_val = 1e9) {
  mjtNum v = 0;
  if (ParseNum(val, &v, min_val, max_val)) {
    out = v;
    return true;
  }
  return false;
}


// helper: parse 3D vector "x,y,z"
static bool ParseVec3(string_view val, optional<std::array<mjtNum, 3>>& out) {
  string str(val);
  double v0, v1, v2;
  if (std::sscanf(str.c_str(), "%lf,%lf,%lf", &v0, &v1, &v2) == 3) {
    out = {static_cast<mjtNum>(v0), static_cast<mjtNum>(v1), static_cast<mjtNum>(v2)};
    return true;
  }
  return false;
}


// helper: parse visualization and rendering flags
static bool ParseVisOrRndFlag(string_view name, string_view val, Options& opt) {
  int bool_val = 0;
  if (!ParseBool(val, &bool_val)) { return false; }
  string norm = NormalizeName(name);
  for (int j = 0; j < mjNVISFLAG; j++) {
    if (norm == NormalizeName(mjVISSTRING[j][0])) {
      opt.vis_flags[j] = bool_val;
      return true;
    }
  }
  for (int j = 0; j < mjNRNDFLAG; j++) {
    if (norm == NormalizeName(mjRNDSTRING[j][0])) {
      opt.rnd_flags[j] = bool_val;
      return true;
    }
  }
  return false;
}


// helper: parse label type
static bool ParseLabel(string_view val, int* out) {
  int int_val = -1;
  if (ParseInt(val, &int_val, 0, mjNLABEL - 1)) {
    *out = int_val;
    return true;
  }
  string norm = NormalizeName(val);
  for (int i = 0; i < mjNLABEL; i++) {
    if (norm == NormalizeName(mjLABELSTRING[i])) {
      *out = i;
      return true;
    }
  }
  return false;
}


// helper: parse frame type
static bool ParseFrame(string_view val, int* out) {
  int int_val = -1;
  if (ParseInt(val, &int_val, 0, mjNFRAME - 1)) {
    *out = int_val;
    return true;
  }
  string norm = NormalizeName(val);
  for (int i = 0; i < mjNFRAME; i++) {
    if (norm == NormalizeName(mjFRAMESTRING[i])) {
      *out = i;
      return true;
    }
  }
  return false;
}


// helper: parse known flags
static bool ParseFlag(string_view name, string_view val, Options& opt) {
  // clang-format off
  if (name == "backend")   { opt.backend = val; return true; }
  if (name == "width")     { return ParseInt(val, &opt.width, 1); }
  if (name == "height")    { return ParseInt(val, &opt.height, 1); }
  if (name == "camera")    { opt.camera = val; return true; }
  if (name == "lookat")    { return ParseVec3(val, opt.lookat); }
  if (name == "distance")  { return ParseNum(val, opt.distance, 0); }
  if (name == "azimuth")   { return ParseNum(val, opt.azimuth); }
  if (name == "elevation") { return ParseNum(val, opt.elevation); }
  if (name == "key")       { opt.key = val; return true; }
  if (name == "steps")     { return ParseInt(val, &opt.steps, 0); }
  if (name == "geomgroup") { opt.geomgroup = val; return true; }
  if (name == "sitegroup") { opt.sitegroup = val; return true; }
  if (name == "label")     { return ParseLabel(val, &opt.label); }
  if (name == "frame")     { return ParseFrame(val, &opt.frame); }
  // clang-format on
  return ParseVisOrRndFlag(name, val, opt);
}

// helper: parse command line arguments
static bool ParseCommandLine(int& argc, char** argv, Options& opt) {

  for (int i = 1; i < argc; i++) {
    string_view arg(argv[i]);
    if (arg.starts_with("--")) {
      arg.remove_prefix(2);
      string_view name;
      string_view val;
      size_t      eq = arg.find('=');
      if (eq != string_view::npos) {
        name = arg.substr(0, eq);
        val  = arg.substr(eq + 1);
      } else {
        name = arg;
        if (i + 1 < argc && argv[i + 1][0] != '-') {
          val = argv[++i];
        } else {
          val = "1";
        }
      }

      if (!ParseFlag(name, val, opt)) {
        printf("Unknown or invalid option: --%.*s\n", static_cast<int>(name.size()), name.data());
        return false;
      }
    } else if (opt.model_path.empty()) {
      opt.model_path = argv[i];
    } else if (opt.output_path.empty()) {
      opt.output_path = argv[i];
    } else {
      printf("Unexpected positional argument: %s\n", argv[i]);
      return false;
    }
  }

  return true;
}


// helper: derive output filename <model_basename>.png from model path
static string DefaultOutputFilename(string_view model_path) {
  return std::filesystem::path(model_path).stem().string() + ".png";
}


//-------------------------------- filament backend (label rendering) ------------------------------

struct LabelItem {
  string text;
  int    x;
  int    y;
};

// draw bitmap text into RGB pixel buffer (row 0 is top)
static void DrawBitmapText(vector<unsigned char>&   rgb,
                           int                      width,
                           int                      height,
                           const vector<LabelItem>& labels) {
  for (const auto& item : labels) {
    int cur_x = item.x;
    int cur_y = item.y;
    for (unsigned char ch : item.text) {
      if (ch < 32 || ch > 126) { continue; }

      int adr = 0;
      for (unsigned char c = 32; c < ch; c++) {
        int w        = font_normal150[adr + 1];
        int h        = font_normal150[adr + 2];
        int w_bytes  = (w - 1) / 8 + 1;
        adr         += 3 + w_bytes * h;
      }

      int char_w  = font_normal150[adr + 1];
      int char_h  = font_normal150[adr + 2];
      int w_bytes = (char_w - 1) / 8 + 1;

      const unsigned char* bmp_normal = font_normal150 + adr + 3;
      const unsigned char* bmp_back   = font_back150 + adr + 3;

      for (int r = 0; r < char_h; r++) {
        int py = cur_y - r;
        if (py < 0 || py >= height) { continue; }

        for (int c = 0; c < char_w; c++) {
          int px = cur_x + c;
          if (px < 0 || px >= width) { continue; }

          int byte_idx = r * w_bytes + (c / 8);
          int bit_mask = 128 >> (c % 8);

          int pixel_idx = (py * width + px) * 3;
          if (bmp_normal[byte_idx] & bit_mask) {
            rgb[pixel_idx]     = 255;
            rgb[pixel_idx + 1] = 255;
            rgb[pixel_idx + 2] = 255;
          } else if (bmp_back[byte_idx] & bit_mask) {
            rgb[pixel_idx]     /= 2;
            rgb[pixel_idx + 1] /= 2;
            rgb[pixel_idx + 2] /= 2;
          }
        }
      }
      cur_x += char_w;
    }
  }
}


//-------------------------------- filament backend (asset resolution) -----------------------------

// directory containing the running executable (assets are deployed alongside it)
static std::string ExecutableDir() {
  char buf[4096];
#if defined(__APPLE__)
  uint32_t size = sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) != 0) { return ""; }
#elif defined(_WIN32)
  DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf)) { return ""; }
#else
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) { return ""; }
  buf[n] = '\0';
#endif
  return std::filesystem::path(buf).parent_path().string();
}


// resolves a "filament:name" resource to <exe_dir>/assets/name
static std::string ResolveAsset(string_view path) {
  string_view           name = path.substr(path.find(':') + 1);
  std::filesystem::path dir  = ExecutableDir();
  return ((dir.empty() ? std::filesystem::path("assets") : dir / "assets") / name).string();
}


// reads a file lazily, holding the bytes until the resource is closed
class FileResource {
 public:
  explicit FileResource(const std::string& path) : file_(path, std::ios::binary | std::ios::ate) {
    if (file_.is_open()) {
      size_ = file_.tellg();
      file_.seekg(0, std::ios::beg);
    }
  }
  int Size() const { return size_; }
  int Read(const void** buffer) {
    buffer_.resize(size_);
    if (!file_.read(buffer_.data(), size_)) { return 0; }
    *buffer = buffer_.data();
    return size_;
  }

 private:
  std::ifstream     file_;
  std::vector<char> buffer_;
  int               size_ = 0;
};


// registers a resource provider so the Filament backend can load its shaders and textures
// (referenced as "filament:name") from the assets deployed next to the executable
static void RegisterFilamentAssetProvider() {

  mjpResourceProvider provider;  // NOLINT
  mjp_defaultProvider(&provider);
  provider.open = [](mjResource* resource) {
    FileResource* f = new FileResource(ResolveAsset(resource->name));
    if (f->Size() == 0) {
      delete f;
      return 0;
    }
    resource->data = f;
    return f->Size();
  };
  provider.read = [](mjResource* resource, const void** buffer) {
    return static_cast<FileResource*>(resource->data)->Read(buffer);
  };
  provider.close = [](mjResource* resource) {
    delete static_cast<FileResource*>(resource->data);
    resource->data = nullptr;
  };
  provider.prefix = "filament";
  mjp_registerProvider(&provider);
}


//-------------------------------- filament backend (rendering) ------------------------------------

// render scene with Filament backend
static vector<unsigned char> RenderFilament(const mjModel*   m,
                                            mjData*          d,
                                            const mjvCamera& cam,
                                            const mjvOption& opt,
                                            const int        rnd_flags[mjNRNDFLAG],
                                            int              width,
                                            int              height) {
  RegisterFilamentAssetProvider();

  mjrfContextConfig context_cfg;
  mjrf_defaultContextConfig(&context_cfg);
  context_cfg.graphics_api = mjGRAPHICS_API_OPENGL;
  auto ctx                 = mujoco::CreateContext(context_cfg);
  if (!ctx) { mju_error("Could not create Filament context"); }

  // create and configure scene
  mjrfSceneParams scene_params;
  mjrf_defaultSceneParams(&scene_params);
  auto scene = mujoco::CreateScene(ctx.get(), scene_params);
  mjrf_configureSceneFromModel(scene.get(), m);

  // create scene components
  auto model_resources   = std::make_unique<ModelResources>(m, ctx.get());
  auto model_lights      = std::make_unique<ModelLights>(scene.get(), model_resources.get());
  auto model_renderables = std::make_unique<ModelRenderables>(scene.get(), model_resources.get());
  auto model_decorations = std::make_unique<ModelDecorations>(ctx.get(), scene.get(), m);

  // 2D text label callback
  vector<LabelItem> labels;
  auto              draw_text = [&](const char* txt, float x, float y, float z) {
    if (z >= -1.0f && z <= 1.0f) {
      int px = static_cast<int>((x * 0.5f + 0.5f) * width);
      int py = static_cast<int>((0.5f - y * 0.5f) * height);
      labels.push_back({txt, px, py});
    }
  };

  // update scene components
  model_renderables->SetOptions(opt);
  model_lights->Update(d);
  model_renderables->Update(d);
  mjrRect   viewport = {0, 0, width, height};
  mjvCamera cam_copy = cam;
  model_decorations->Update(d, &opt, nullptr, &cam_copy, viewport, draw_text);

  // create render target
  mjrfRenderTargetConfig target_cfg;
  mjrf_defaultRenderTargetConfig(&target_cfg);
  target_cfg.width        = width;
  target_cfg.height       = height;
  target_cfg.color_format = mjPIXEL_FORMAT_RGB8;
  target_cfg.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  auto render_target      = mujoco::CreateRenderTarget(ctx.get(), target_cfg);

  // create render request
  mjrfRenderRequest request;
  mjrf_defaultRenderRequest(&request);
  request.scene               = scene.get();
  request.camera              = mjv_camera2GLCamera(m, d, &cam);
  request.viewport            = viewport;
  request.target              = render_target.get();
  request.headlight           = m->vis.headlight.active;
  request.headlight_color[0]  = m->vis.headlight.diffuse[0];
  request.headlight_color[1]  = m->vis.headlight.diffuse[1];
  request.headlight_color[2]  = m->vis.headlight.diffuse[2];
  request.headlight_intensity = model_lights->GetHeadlightIntensity();

  // process render flags
  if (rnd_flags[mjRND_SEGMENT] > 0) {
    if (rnd_flags[mjRND_IDCOLOR] > 0) {
      request.draw_mode = mjDRAW_MODE_SEGMENTATION_BY_ID;
    } else {
      request.draw_mode = mjDRAW_MODE_SEGMENTATION_BY_COLOR;
    }
  } else if (rnd_flags[mjRND_DEPTH] > 0) {
    request.draw_mode = mjDRAW_MODE_DEPTH;
  } else if (rnd_flags[mjRND_WIREFRAME] > 0) {
    request.draw_mode = mjDRAW_MODE_WIREFRAME;
  } else if (opt.flags[mjVIS_ISLAND]) {
    request.draw_mode = mjDRAW_MODE_ISLANDS;
  }
  if (rnd_flags[mjRND_SHADOW] >= 0) { request.shadows = rnd_flags[mjRND_SHADOW]; }
  if (rnd_flags[mjRND_REFLECTION] >= 0) { request.reflections = rnd_flags[mjRND_REFLECTION]; }
  if (rnd_flags[mjRND_SKYBOX] >= 0) { request.skybox = rnd_flags[mjRND_SKYBOX]; }
  if (rnd_flags[mjRND_HAZE] >= 0) { request.haze = rnd_flags[mjRND_HAZE]; }
  if (rnd_flags[mjRND_POST_PROCESSING] >= 0) {
    request.post_processing = rnd_flags[mjRND_POST_PROCESSING];
  }

  // prepare output buffer
  vector<unsigned char> rgb(width * height * 3);

  // prepare read pixels request
  mjrfReadRequest read_request;
  mjrf_defaultReadRequest(&read_request);
  read_request.target    = render_target.get();
  read_request.output    = rgb.data();
  read_request.num_bytes = width * height * 3;

  // render and wait for frame
  const mjrfFrameHandle frame = mjrf_render(ctx.get(), &request, 1, &read_request, 1);
  mjrf_waitForFrame(ctx.get(), frame);

  // draw text labels
  DrawBitmapText(rgb, width, height, labels);

  // clear decorations before destroying scene
  model_decorations->Clear();

  // return pixels
  return rgb;
}


//-------------------------------- classic backend -------------------------------------------------

// render scene with classic OpenGL backend
static vector<unsigned char> RenderClassic(mjModel*         m,
                                           mjData*          d,
                                           const mjvCamera& cam,
                                           const mjvOption& opt,
                                           const int        rnd_flags[mjNRNDFLAG],
                                           int              width,
                                           int              height) {
  // create scene
  static constexpr int kMaxGeom = 50000;
  mjvScene             scn;
  mjv_defaultScene(&scn);
  mjv_makeScene(m, &scn, kMaxGeom);

  // apply custom render flags
  for (int i = 0; i < mjNRNDFLAG; i++) {
    if (rnd_flags[i] >= 0) { scn.flags[i] = rnd_flags[i]; }
  }

  // update offscreen buffer size to match requested resolution
  m->vis.global.offwidth  = width;
  m->vis.global.offheight = height;

  // create and configure rendering context
  mjrContext con;
  mjr_defaultContext(&con);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // update scene and render
  mjvCamera cam_copy = cam;
  mjv_updateScene(m, d, &opt, nullptr, &cam_copy, mjCAT_ALL, &scn);
  mjrRect viewport = {0, 0, width, height};
  mjr_setBuffer(mjFB_OFFSCREEN, &con);
  mjr_render(viewport, &scn, &con);

  // read pixels from offscreen buffer
  vector<unsigned char> rgb(3 * width * height);
  mjr_readPixels(rgb.data(), nullptr, viewport, &con);

  // free resources
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // flip image vertically in-place
  for (int r = 0; r < height / 2; r++) {
    std::swap_ranges(rgb.begin() + r * 3 * width,
                     rgb.begin() + (r + 1) * 3 * width,
                     rgb.begin() + (height - 1 - r) * 3 * width);
  }

  // return pixels
  return rgb;
}


//-------------------------------- main ------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("%s", help_msg);
    return EXIT_SUCCESS;
  }

  for (int i = 1; i < argc; i++) {
    string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      printf("%s", help_msg);
      return EXIT_SUCCESS;
    }
  }

  Options opt;
  if (!ParseCommandLine(argc, argv, opt)) { return EXIT_FAILURE; }

  if (opt.model_path.empty()) {
    printf("%s", help_msg);
    return EXIT_FAILURE;
  }

  // determine output filename if not provided
  string output_file =
      opt.output_path.empty() ? DefaultOutputFilename(opt.model_path) : opt.output_path;

  // load model
  char error[1000] = "Could not load binary model";

  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> m(
      opt.model_path.ends_with(".mjb")
          ? mj_loadModel(opt.model_path.c_str(), nullptr)
          : mj_loadXML(opt.model_path.c_str(), nullptr, error, sizeof(error)),
      mj_deleteModel);
  if (!m) {
    printf("Load model error: %s\n", error);
    return EXIT_FAILURE;
  }

  std::unique_ptr<mjData, decltype(&mj_deleteData)> d(mj_makeData(m.get()), mj_deleteData);
  if (!d) {
    printf("Could not allocate mjData\n");
    return EXIT_FAILURE;
  }

  // set keyframe state if requested or if default "test" keyframe exists
  int key_id = -1;
  if (!opt.key.empty()) {
    key_id = mj_name2id(m.get(), mjOBJ_KEY, opt.key.c_str());
    if (key_id < 0) {
      int parsed = -1;
      if (ParseInt(opt.key, &parsed, 0, m->nkey - 1)) {
        key_id = parsed;
      } else {
        printf("Keyframe not found: %s\n", opt.key.c_str());
        return EXIT_FAILURE;
      }
    }
  } else {
    key_id = mj_name2id(m.get(), mjOBJ_KEY, "test");
  }

  // load keyframe
  if (key_id >= 0) { mj_resetDataKeyframe(m.get(), d.get(), key_id); }

  // advance the simulation before rendering
  for (int i = 0; i < opt.steps; i++) { mj_step(m.get(), d.get()); }

  // run forward dynamics
  mj_forward(m.get(), d.get());

  // configure camera
  mjvCamera cam;
  mjv_defaultFreeCamera(m.get(), &cam);
  if (!opt.camera.empty()) {
    int cam_id = mj_name2id(m.get(), mjOBJ_CAMERA, opt.camera.c_str());
    if (cam_id < 0) {
      int parsed = -1;
      if (ParseInt(opt.camera, &parsed, 0, m->ncam - 1)) {
        cam_id = parsed;
      } else if (opt.camera != "free" && opt.camera != "default") {
        printf("Camera not found: %s\n", opt.camera.c_str());
        return EXIT_FAILURE;
      }
    }
    if (cam_id >= 0) {
      cam.type       = mjCAMERA_FIXED;
      cam.fixedcamid = cam_id;
    }
  }

  // free camera overrides
  if (cam.type == mjCAMERA_FREE) {
    if (opt.lookat) mju_copy3(cam.lookat, opt.lookat->data());
    if (opt.distance) cam.distance = *opt.distance;
    if (opt.azimuth) cam.azimuth = *opt.azimuth;
    if (opt.elevation) cam.elevation = *opt.elevation;
  }

  // configure visualization options
  mjvOption vopt;
  mjv_defaultOption(&vopt);

  // geom and site groups
  auto set_group = [](string_view opt_val, const char* name, unsigned char* group) {
    if (opt_val.empty()) return true;
    if (opt_val.size() != mjNGROUP) {
      printf("Invalid --%s argument, expected %d digits (0 or 1)\n", name, mjNGROUP);
      return false;
    }
    for (int i = 0; i < mjNGROUP; i++) group[i] = (opt_val[i] == '1');
    return true;
  };

  if (!set_group(opt.geomgroup, "geomgroup", vopt.geomgroup) ||
      !set_group(opt.sitegroup, "sitegroup", vopt.sitegroup)) {
    return EXIT_FAILURE;
  }

  // vis flags
  for (int i = 0; i < mjNVISFLAG; i++) {
    if (opt.vis_flags[i] >= 0) { vopt.flags[i] = opt.vis_flags[i]; }
  }

  // label and frame
  if (opt.label >= 0) { vopt.label = opt.label; }
  if (opt.frame >= 0) { vopt.frame = opt.frame; }

  // render image
  vector<unsigned char> rgb;
  if (opt.backend == "filament") {
    rgb = RenderFilament(m.get(), d.get(), cam, vopt, opt.rnd_flags, opt.width, opt.height);
  } else if (opt.backend == "classic") {
    rgb = RenderClassic(m.get(), d.get(), cam, vopt, opt.rnd_flags, opt.width, opt.height);
  } else {
    printf("Unknown backend: %s (expected 'filament' or 'classic')\n", opt.backend.c_str());
    return EXIT_FAILURE;
  }

  // encode and write PNG
  unsigned error_code =
      lodepng_encode24_file(output_file.c_str(), rgb.data(), opt.width, opt.height);
  if (error_code) {
    printf("Error saving PNG %s: %s\n", output_file.c_str(), lodepng_error_text(error_code));
    return EXIT_FAILURE;
  }

  printf("Saved %dx%d image to %s\n", opt.width, opt.height, output_file.c_str());
  return EXIT_SUCCESS;
}
