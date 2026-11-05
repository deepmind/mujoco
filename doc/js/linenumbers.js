// DO NOT EDIT. THE SRCS ARRAY IS AUTOMATICALLY GENERATED.
const SRCS = [
  'batch/batch.cc',
  'engine/engine_callback.c',
  'engine/engine_collision_driver.c',
  'engine/engine_collision_sdf.c',
  'engine/engine_core_constraint.c',
  'engine/engine_core_smooth.c',
  'engine/engine_core_util.c',
  'engine/engine_derivative.c',
  'engine/engine_derivative_fd.c',
  'engine/engine_forward.c',
  'engine/engine_init.c',
  'engine/engine_inverse.c',
  'engine/engine_io.c',
  'engine/engine_island.c',
  'engine/engine_memory.c',
  'engine/engine_metric.c',
  'engine/engine_name.c',
  'engine/engine_passive.c',
  'engine/engine_plugin.cc',
  'engine/engine_print.c',
  'engine/engine_ray.c',
  'engine/engine_sensor.c',
  'engine/engine_setconst.c',
  'engine/engine_sleep.c',
  'engine/engine_support.c',
  'engine/engine_thread.cc',
  'engine/engine_util_blas.c',
  'engine/engine_util_errmem.c',
  'engine/engine_util_misc.c',
  'engine/engine_util_solve.c',
  'engine/engine_util_sparse.c',
  'engine/engine_util_spatial.c',
  'engine/engine_vis_init.c',
  'engine/engine_vis_interact.c',
  'engine/engine_vis_visualize.c',
  'experimental/filament/mjr_compat.cc',
  'render/classic/render_context.c',
  'render/classic/render_gl2.c',
  'render/classic/render_gl3.c',
  'render/classic/render_util.c',
  'render/filament/mjrfilament.cc',
  'render/noop/render_noop.c',
  'ui/ui_main.c',
  'user/user_api.cc',
  'user/user_init.c',
  'user/user_resource.cc',
  'user/user_vfs.cc',
  'xml/xml_api.cc',
  'xml/xml_util.cc',
];

class LineNumbers {
  constructor() {
    this.map = new Map();
  }

  static fetch(src) {
    const url = `https://raw.githubusercontent.com/google-deepmind/mujoco/refs/heads/main/src/${src}`;
    const request = new XMLHttpRequest();
    return new Promise((resolve, reject) => {
      request.onreadystatechange = () => {
        if (request.readyState === 4) {
          if (request.status === 200) {
            resolve(request.responseText);
          } else {
            reject(request.status);
          }
        }
      };
      request.open('GET', url, true);
      request.send();
    });
  }

  fetchAll() {
    let requests = [];
    for (const src of SRCS) {
      requests.push(LineNumbers.fetch(src).then(contents => {
        this.processSrc(src, contents);
      }, reason => {/* swallow error */}));
    }
    Promise.all(requests).then(() => {
      const anchors = document.querySelectorAll('h3 a.reference.external');
      for (const anchor of anchors) {
        const url = anchor.getAttribute('href');
        if (url.startsWith('#')) {
          const key = url.substring(1);
          if (this.map.has(key)) {
            anchor.href = this.map.get(key);
          } else {
            console.log(`No line number found for ${key}`);
          }
        }
      }
    });
  }

  processSrc(src, contents) {
    const lines = contents.split('\n');
    const re = /^(const )?[a-zA-Z0-9_*]+\s(.+)\(.+[{,]$/;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].match(re)) {
        const key = lines[i].match(re)[2];
        this.map.set(key, `https://github.com/google-deepmind/mujoco/blob/main/src/${src}#L${i+1}`);
      }
    }

    // edge cases
    if (src == 'user/user_api.cc') {
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('[[nodiscard]] int mj_recompile(')) {
          const key = 'mj_recompile';
          this.map.set(key, `https://github.com/google-deepmind/mujoco/blob/main/src/${src}#L${i+1}`);
        }
      }
    } else if (src == 'engine/engine_io.c') {
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('void mj_freeStack(')) {
          const key = 'mj_freeStack';
          this.map.set(key, `https://github.com/google-deepmind/mujoco/blob/main/src/${src}#L${i+1}`);
        } else if (lines[i].startsWith('void mj_markStack(')) {
          const key = 'mj_markStack';
          this.map.set(key, `https://github.com/google-deepmind/mujoco/blob/main/src/${src}#L${i+1}`);
        }
      }
    }
  }
}

window.onload = () => {
  if (document.getElementById('fetchlines')) {
    let lines = new LineNumbers();
    lines.fetchAll();
  }
};
