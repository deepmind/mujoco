// Copyright 2026 Google LLC
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

// --- Default model redirect ---
// When no ?model= parameter is present, redirect to the same page with the
// default model URL. This fires before WASM starts loading, so no time is
// wasted. Uses replace() to avoid polluting browser history.
// Visit ?model= (empty) to explicitly get an empty scene without redirect.
{
  const params = new URLSearchParams(window.location.search);
  if (!params.has('model')) {
    const defaultModel =
        'github:google-deepmind/mujoco/main/model/welcome/welcome.xml';
    // Construct the query string manually instead of using URLSearchParams.toString()
    // or encodeURIComponent() so that ':' and '/' in defaultModel remain unencoded
    // and human-readable in the browser's address bar.
    const existing = window.location.search.replace(/^\?/, '');
    const query = (existing ? existing + '&' : '') + 'model=' + defaultModel;
    window.location.replace(
        window.location.pathname + '?' + query + window.location.hash);
    // Stop all further execution while the browser navigates.
    throw new Error('Redirecting to default model');
  }
}

// --- Loading overlay (shown while model is compiling) ---
const loadingOverlay = document.getElementById('loadingOverlay');
function showLoading() {
  loadingOverlay.style.display = 'flex';
}
function hideLoading() {
  loadingOverlay.style.display = 'none';
}

function isIOSDevice() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

// iOS WebKit has a substantially lower WebAssembly memory ceiling than
// desktop browsers. Select the single-threaded, lower-memory build before the
// Emscripten runtime creates its WebAssembly.Memory object.
const wasmRuntimeDirectory = isIOSDevice() ? 'bin/ios' : 'bin';

// ---------------------------------------------------------------------------
// Parallel asset prefetcher.
//
// MuJoCo's XML compiler synchronously opens each referenced asset via
// mjpResourceProvider. In WASM that becomes a chain of ASYNCIFY-suspended
// fetch() calls, strictly serial, one RTT per asset. For Menagerie models
// that's ~80 round trips and ~10 s of avoidable wall time on a cold cache.
//
// To sidestep this we let MuJoCo itself enumerate the dependencies (via
// mju_getXMLDependencies, exposed to JS as Module.getXMLDependencies), then
// fetch them all in parallel and feed the bytes into the WASM-side
// FetchCache via Module.primeFetchCache. When Module.loadUrl subsequently
// runs, every resource the compiler opens is already in memory.
//
// Discovery rules live in src/xml/xml_util.cc; the JS only has to know
// the scheme rewrites used by the resource providers in emscripten.cc.
// ---------------------------------------------------------------------------

// MuJoCo Live custom URL schemes that the C++ resource providers rewrite
// before fetching. JS must do the same so primed cache keys match what
// the providers eventually request.
function resolveScheme(url) {
  if (url.startsWith('github:')) {
    return 'https://raw.githubusercontent.com/' + url.slice('github:'.length);
  }
  return url;
}

async function prefetchModelAssets(rootUrl, onProgress) {
  // Only prefetch when the root URL is one we know how to fetch from
  // Javascript. Other resource-provider-backed schemes (e.g., uploaded files
  // via the drag-and-drop path) skip the prefetcher entirely and let
  // Module.loadUrl handle take the existing slow path.
  if (!/^(https?:|github:)/.test(rootUrl)) {
    return { files: 0, bytes: 0, errors: 0 };
  }

  const primed = new Set();   // URLs we've already pushed into FetchCache
  const stats = { files: 0, bytes: 0, errors: 0 };

  // Fetch + prime, dedup against `primed`. Returns true on success.
  async function fetchAndPrime(httpUrl) {
    if (primed.has(httpUrl)) return true;
    primed.add(httpUrl);
    try {
      const resp = await fetch(httpUrl);
      if (!resp.ok) {
        console.warn('[prefetch] HTTP', resp.status, httpUrl);
        stats.errors++;
        return false;
      }
      const bytes = new Uint8Array(await resp.arrayBuffer());
      Module.primeFetchCache(httpUrl, bytes);
      stats.files++;
      stats.bytes += bytes.length;
      onProgress?.(stats);
      return true;
    } catch (err) {
      console.warn('[prefetch]', httpUrl, err);
      stats.errors++;
      return false;
    }
  }

  // Step 1: fetch the root XML so mju_getXMLDependencies (which reads it
  // through the resource provider) hits cache instead of the network.
  // Also gives the loading screen an immediate progress tick.
  const rootHttp = resolveScheme(rootUrl);
  if (!await fetchAndPrime(rootHttp)) {
    throw new Error(`could not fetch root model ${rootHttp}`);
  }

  // Step 2: ask MuJoCo for the transitive dependency list.
  // When rootUrl points to a plain MJCF XML, mju_getXMLDependencies parses the
  // XML text to find external dependencies (meshes, textures, includes) so they
  // can be pre-fetched in parallel.
  // If rootUrl is a binary archive (.mjz, .zip, or an extensionless HTTP asset),
  // getXMLDependencies cannot parse the binary payload as XML and may throw or
  // fail. Since binary archives are completely self-contained, no external URL
  // prefetching is needed, and we safely catch the error and continue.
  try {
    const depsVec = await Module.getXMLDependencies(rootUrl);
    if (depsVec) {
      const deps = [];
      for (let i = 0; i < depsVec.size(); ++i) deps.push(depsVec.get(i));
      depsVec.delete();
      // Step 3: fetch every dependency in parallel, deduped against URLs
      // already primed in step 1 (and earlier deps within this batch).
      await Promise.all(deps.map((depUrl) => fetchAndPrime(resolveScheme(depUrl))));
    }
  } catch (e) {
    // Binary archives (.mjz, .zip) have self-contained dependencies and do not
    // require XML dependency prefetching.
    console.log('[prefetch] Self-contained archive or non-XML root model:', rootUrl);
  }
  return stats;
}

var Module = {
  preRun: [],
  postRun: [],
  locateFile: function (path) {
    const baseURL = window.location.origin + window.location.pathname.substring(0, window.location.pathname.lastIndexOf("/"));
    return baseURL + "/" + wasmRuntimeDirectory + "/" + path;
  },
  print: console.log,
  printErr: text => {
    console.error(text + "\n" + new Error().stack);
  },
  canvas: (() => {
    const canvas = document.getElementById("canvas");
    // As a default initial behavior, pop up an alert when webgl context is lost. To make your
    // application robust, you may want to override this behavior before shipping!
    // See http://www.khronos.org/registry/webgl/specs/latest/1.0/#5.15.2
    canvas.addEventListener(
      "webglcontextlost",
      e => {
        alert("WebGL context lost. You will need to reload the page.");
        e.preventDefault();
      },
      false,
    );
    return canvas;
  })(),
  setStatus(text) { },
  totalDependencies: 0,
  monitorRunDependencies(left) { },
  onRuntimeInitialized: () => {
    const assetsToPrefetch = [
      "assets/fontawesome-webfont.ttf",
      "assets/ibl.ktx",
      "assets/AtkinsonHyperlegibleNext[wght].ttf",
      "assets/AtkinsonHyperlegibleMono-Regular.ttf",
      "assets/pbr.filamat",
      "assets/pbr_reflect.filamat",
      "assets/pbr_packed_reflect.filamat",
      "assets/pbr_transparent.filamat",
      "assets/pbr_packed.filamat",
      "assets/pbr_packed_transparent.filamat",
      "assets/phong_2d_fade.filamat",
      "assets/phong_2d.filamat",
      "assets/phong_2d_uv_fade.filamat",
      "assets/phong_2d_uv.filamat",
      "assets/phong_2d_uv_reflect.filamat",
      "assets/phong_color_fade.filamat",
      "assets/phong_color.filamat",
      "assets/phong_color_reflect.filamat",
      "assets/phong_cube_fade.filamat",
      "assets/phong_cube.filamat",
      "assets/phong_cube_reflect.filamat",
      "assets/outline_composite.filamat",
      "assets/outline_flatten.filamat",
      "assets/outline_jumpflood.filamat",
      "assets/decor.filamat",
      "assets/decor_fade.filamat",
      "assets/haze.filamat",
      "assets/unlit_depth.filamat",
      "assets/unlit_segmentation.filamat",
      "assets/unlit_ui.filamat"
    ];

    const assetPromises = assetsToPrefetch.map(async (relativePath) => {
      const assetUrl = Module.locateFile(relativePath);
      try {
        const response = await fetch(assetUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${assetUrl}: ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();
        const filename = relativePath.substring(relativePath.lastIndexOf('/') + 1);
        Module.registerAsset(filename, new Uint8Array(buffer));
        console.log(`Registered asset: ${filename}`);
      } catch (error) {
        console.error(`Error prefetching asset ${assetUrl}:`, error);
        throw error;  // Re-throw to be caught by Promise.all
      }
    });

    Promise.all(assetPromises)
      .then(() => {
        try {
          const prefersDark = true;
          Module.init("MuJoCo Live", prefersDark);

          // Check for ?model= and ?keyframe= URL parameters and load from URL.
          const params = new URLSearchParams(window.location.search);
          const modelUrl = params.get('model');
          const keyframe = params.get('keyframe');
          if (modelUrl) {
            // loadUrl uses ASYNCIFY (via EM_ASYNC_JS fetch), which
            // suspends the WASM module. We must not start the animation
            // loop until the load completes, otherwise renderFrame will
            // hit "Cannot have multiple async operations in flight".
            showLoading();
            // Surface prefetch progress on the loading screen.
            const dlEl = document.getElementById('loadingDownload');
            const dlFileEl = document.getElementById('loadingDownloadFile');
            if (dlEl) dlEl.style.display = 'block';
            const onProgress = (s) => {
              if (dlFileEl) {
                dlFileEl.textContent =
                    `${s.files} files · ${(s.bytes/1024/1024).toFixed(1)} MB`;
              }
            };
            requestAnimationFrame(() => {
              requestAnimationFrame(async () => {
                try {
                  await prefetchModelAssets(modelUrl, onProgress);
                  await Module.loadUrl(modelUrl);
                  if (keyframe !== null) {
                    Module.loadKeyframe(keyframe);
                  }
                } catch (error) {
                  console.error('Failed to load model from URL:', error);
                } finally {
                  if (dlEl) dlEl.style.display = 'none';
                  hideLoading();
                  requestAnimationFrame(Module.animate);
                }
              });
            });
          } else {
            requestAnimationFrame(Module.animate);
          }
        } catch (error) {
          console.error('Failed to initialize app.', error);
        }
      })
      .catch((error) => {
        console.error('Failed to prefetch one or more assets.', error);
      });
  },
  animate: async () => {
    try {
      await Module.renderFrame();
    } catch (error) {
      console.error('Update error:', error);
    }
    requestAnimationFrame(Module.animate);
  },
};
// Ensure the canvas is resized when the window is resized.
window.addEventListener("resize", function () {
  Module.canvas.style.width = window.innerWidth + "px";
  Module.canvas.style.height = window.innerHeight + "px";
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const buffer = e.target.result;
    // Show the loading overlay, then defer the blocking WASM call by two
    // animation frames so the browser has a chance to paint the overlay.
    showLoading();
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        try {
          Module.loadFile(file.name, buffer);
        } catch (error) {
          console.error('Failed to load model from file:', error);
        } finally {
          hideLoading();
        }
      });
    });
  };
  reader.onerror = (e) => {
    console.error('Error reading file:', e);
  };
  reader.readAsArrayBuffer(file);
}

document.addEventListener('DOMContentLoaded', () => {
  // Remove commit hash if not substituted
  var el = document.getElementById('commitHash');
  if (el && el.textContent === '__COMMIT_HASH_PLACEHOLDER__') el.remove();

  // --- Drag-and-drop support ---
  const dropOverlay = document.getElementById('dropOverlay');

  let dragCounter = 0;

  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    if (dragCounter === 1) {
      dropOverlay.style.display = 'flex';
    }
  });

  document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter === 0) {
      dropOverlay.style.display = 'none';
    }
  });

  document.addEventListener('dragover', (e) => {
    e.preventDefault();
  });

  document.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropOverlay.style.display = 'none';
    const files = e.dataTransfer.files;
    for (let i = 0; i < files.length; i++) {
      handleFile(files[i]);
    }
  });
});

const wasmRuntimeScript = document.createElement('script');
wasmRuntimeScript.async = true;
wasmRuntimeScript.src = Module.locateFile('mujoco_studio.js');
wasmRuntimeScript.onerror = () => {
  hideLoading();
  console.error('Failed to load MuJoCo Live runtime:', wasmRuntimeScript.src);
};
document.body.appendChild(wasmRuntimeScript);
