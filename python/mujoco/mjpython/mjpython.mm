// Copyright 2023 DeepMind Technologies Limited
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

#import <algorithm>
#import <atomic>
#import <cstdint>
#import <cstdlib>
#import <cstring>
#import <iostream>
#import <vector>

#import <dlfcn.h>
#import <mach-o/dyld.h>
#import <pthread.h>
#import <sys/resource.h>
#import <unistd.h>

#import <Cocoa/Cocoa.h>
#import <Python.h>

extern "C" {
extern char **environ;  // for execve

// Wrap Objective-C Cocoa calls into C-style functions with default visibility,
// so that we can dlsym and call them from Python via ctypes.
__attribute__((used)) void mjpython_hide_dock_icon() {
  [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

  // Drain pending Cocoa events on the main thread. The activation policy
  // change triggers an asynchronous dock/screen reconfiguration via
  // SkyLight. If we return before it's processed, the notification lands
  // on a GCD worker thread and hits NSScreen's main-thread assertion.
  NSDate* soon = [NSDate dateWithTimeIntervalSinceNow:0.01];
  [[NSRunLoop mainRunLoop] runUntilDate:soon];
}
__attribute__((used)) void mjpython_show_dock_icon() {
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
}
}  // extern "C"

namespace {
// 16MiB is the default Python thread stack size on macOS as of Python 3.11
// https://bugs.python.org/issue18075
constexpr rlim_t kThreadStackSize = 0x1000000;

struct {
#define CPYTHON_FN(fname) decltype(&::fname) fname

  // go/keep-sorted start
  CPYTHON_FN(PyConfig_Clear);
  CPYTHON_FN(PyConfig_InitPythonConfig);
  CPYTHON_FN(PyConfig_SetBytesArgv);
  CPYTHON_FN(PyGILState_Ensure);
  CPYTHON_FN(PyGILState_Release);
  CPYTHON_FN(PyRun_SimpleStringFlags);
  CPYTHON_FN(PyStatus_Exception);
  CPYTHON_FN(Py_ExitStatusException);
  CPYTHON_FN(Py_FinalizeEx);
  CPYTHON_FN(Py_InitializeFromConfig);
  CPYTHON_FN(Py_RunMain);
  // go/keep-sorted end

#undef CPYTHON_FN
} cpython;

std::atomic_bool py_initialized;

struct Args {
  int argc;
  char** argv;
};

// The Python main thread (distinct from the macOS main thread, which executes the main function).
void* mjpython_pymain(void* vargs) {
  Args* args = static_cast<Args*>(vargs);
  PyGILState_STATE gil;

  // Initialize the Python interpreter.
  PyConfig config;
  cpython.PyConfig_InitPythonConfig(&config);
  {
    PyStatus status = cpython.PyConfig_SetBytesArgv(&config, args->argc, args->argv);
    if (cpython.PyStatus_Exception(status)) {
      // Calls exit() which aborts the entire process.
      cpython.Py_ExitStatusException(status);
    }
  }
  {
    PyStatus status = cpython.Py_InitializeFromConfig(&config);
    if (cpython.PyStatus_Exception(status)) {
      // Calls exit() which aborts the entire process.
      cpython.Py_ExitStatusException(status);
    }
  }
  cpython.PyConfig_Clear(&config);

  // Set up the condition variable to pass control back to the macOS main thread.
  gil = cpython.PyGILState_Ensure();
  cpython.PyRun_SimpleStringFlags(R"(
def _mjpython_make_cond():
  # Don't pollute the global namespace.
  global _mjpython_make_cond
  del _mjpython_make_cond

  import threading

  global cond
  cond = threading.Condition()

_mjpython_make_cond()
)", nullptr);
  py_initialized.store(true);

  // Wait until GLFW is initialized on macOS main thread, set up the queue and an atexit hook
  // to enqueue a termination flag upon exit.
  cpython.PyRun_SimpleStringFlags(R"(
def _mjpython_init():
  # Don't pollute the global namespace.
  global _mjpython_init
  del _mjpython_init

  import atexit
  import threading

  # The mujoco.viewer module should only be imported after glfw.init() in the macOS main thread.
  with cond:
    cond.wait()
  import mujoco.viewer

  # Similar to a queue.Queue(maxsize=1), but where only one active task is allowed at a time.
  # With queue.Queue(1), another item is allowed to be enqueued before task_done is called.
  class _MjPythonImpl(mujoco.viewer._MjPythonBase):

    # Termination statuses
    NOT_TERMINATED = 0
    TERMINATION_REQUESTED = 1
    TERMINATION_ACCEPTED = 2
    TERMINATED = 3

    def __init__(self):
      self._cond = threading.Condition()
      self._task = None
      self._termination = self.__class__.NOT_TERMINATED
      self._busy = False

    def launch_on_ui_thread(
        self,
        model,
        data,
        handle_return,
        key_callback,
        show_left_ui,
        show_right_ui,
    ):
      with self._cond:
        if self._busy or self._task is not None:
          raise RuntimeError('another MuJoCo viewer is already open')
        else:
          self._task = (
              model,
              data,
              handle_return,
              key_callback,
              show_left_ui,
              show_right_ui,
          )
          self._cond.notify()

    def terminate(self):
      with self._cond:
        self._termination = self.__class__.TERMINATION_REQUESTED
        self._cond.notify()
        self._cond.wait_for(
            lambda: self._termination == self.__class__.TERMINATED)

    def get(self):
      with self._cond:
        self._cond.wait_for(
            lambda: self._task is not None or self._termination)

        if self._termination:
          if self._termination == self.__class__.TERMINATION_REQUESTED:
            self._termination = self.__class__.TERMINATION_ACCEPTED
          return None

        task = self._task
        self._busy = True
        self._task = None
        return task

    def done(self):
      with self._cond:
        self._busy = False
        if self._termination == self.__class__.TERMINATION_ACCEPTED:
          self._termination = self.__class__.TERMINATED
        self._cond.notify()


  mujoco.viewer._MJPYTHON = _MjPythonImpl()
  atexit.register(mujoco.viewer._MJPYTHON.terminate)

  with cond:
    cond.notify()

_mjpython_init()
)", nullptr);

  // Run the Python interpreter main loop.
  cpython.Py_RunMain();

  // Tear down the interpreter.
  cpython.Py_FinalizeEx();
  return nullptr;
}
}  // namespace

int main(int argc, char** argv) {
  py_initialized.store(false);
  const char* libpython_path = getenv("MJPYTHON_LIBPYTHON");
  if (!libpython_path || !libpython_path[0]) {
    std::cerr << "This binary must be launched via the mjpython.py script.\n";
    return EXIT_FAILURE;
  }

  // Enlarge the stack if necessary to match what Python normally expects to have when launching
  // a new thread.
  rlimit stack;
  if (getrlimit(RLIMIT_STACK, &stack)) {
    std::cerr << "getrlimit failed to query stack size with error code " << errno << " ("
              << std::strerror(errno) << ")\n";
    std::cerr << "continuing anyway but crashes may occur if the stack is too small\n";
  } else if (stack.rlim_cur < kThreadStackSize && stack.rlim_cur < stack.rlim_max) {
    auto rlim_old = stack.rlim_cur;
    stack.rlim_cur = std::min(kThreadStackSize, stack.rlim_max);
    if (setrlimit(RLIMIT_STACK, &stack)) {
      std::cerr << "setrlimit failed to increase stack size with error code " << errno << " ("
                << std::strerror(errno) << ")\n";
      std::cerr << "continuing anyway with stack size " << rlim_old << " but crashes may occur\n";
    } else {
      // re-exec the binary so that the new stack size takes effect
      std::uint32_t path_size = 0;
      _NSGetExecutablePath(nullptr, &path_size);
      std::vector<char> path(path_size);
      if (_NSGetExecutablePath(path.data(), &path_size)) {
        std::cerr << "unexpected error from _NSGetExecutablePath, continuing anyway\n";
      } else {
        execve(path.data(), argv, environ);
      }
    }
  }

  // Resolve libpython at runtime to prevent linking against the wrong dylib. The correct libpython
  // path is passed from a Python trampoline script, which ran inside the desired interpreter and
  // exec'd this binary.
  void* libpython = dlopen(libpython_path, RTLD_NOW | RTLD_GLOBAL);
  if (!libpython) {
    std::cerr << "failed to dlopen path '" << libpython_path << "': " << dlerror() << "\n";
    return EXIT_FAILURE;
  }

  // Look up required CPython API functions from table of symbols already loaded into the process.
#define CPYTHON_INITFN(fname)                                                            \
  {                                                                                      \
    cpython.fname = reinterpret_cast<decltype(cpython.fname)>(dlsym(libpython, #fname)); \
    if (!cpython.fname) {                                                                \
      std::cerr << "failed to dlsym '" << #fname << "': " << dlerror() << "\n";          \
      return EXIT_FAILURE;                                                               \
    }                                                                                    \
  }

  // go/keep-sorted start
  CPYTHON_INITFN(PyConfig_Clear);
  CPYTHON_INITFN(PyConfig_InitPythonConfig);
  CPYTHON_INITFN(PyConfig_SetBytesArgv);
  CPYTHON_INITFN(PyGILState_Ensure);
  CPYTHON_INITFN(PyGILState_Release);
  CPYTHON_INITFN(PyRun_SimpleStringFlags);
  CPYTHON_INITFN(PyStatus_Exception);
  CPYTHON_INITFN(Py_ExitStatusException);
  CPYTHON_INITFN(Py_FinalizeEx);
  CPYTHON_INITFN(Py_InitializeFromConfig);
  CPYTHON_INITFN(Py_RunMain);
  // go/keep-sorted end

#undef CPYTHON_INITFN

  // Package up argc and argv together to pass to pthread_create.
  Args args{argc, argv};

#define PTHREAD_CHECKED(func, ...)                                                              \
  {                                                                                             \
    int result = func(__VA_ARGS__);                                                             \
    if (result) {                                                                               \
      std::cerr << #func << " failed with " << result << "(" << std::strerror(result) << ")\n"; \
      return EXIT_FAILURE;                                                                      \
    }                                                                                           \
  }

  // Configure the new thread with the correct stack size;
  pthread_attr_t pthread_attr;
  PTHREAD_CHECKED(pthread_attr_init, &pthread_attr);
  PTHREAD_CHECKED(pthread_attr_setstacksize, &pthread_attr, stack.rlim_cur);

  // Create a thread to be used as the "Python main thread".
  pthread_t pymain_thread;
  PTHREAD_CHECKED(pthread_create, &pymain_thread, &pthread_attr, &mjpython_pymain, &args);
  pthread_attr_destroy(&pthread_attr);
#undef PTHREAD_CHECKED

  // Busy-wait until Python interpreter is initialized.
  while (!py_initialized.load()) {}

  // Initialize GLFW on the macOS main thread, yield control to Python main thread and wait for it
  // to finish setting up _MJPYTHON, then serve incoming viewer launch requests.
  PyGILState_STATE gil = cpython.PyGILState_Ensure();
  cpython.PyRun_SimpleStringFlags(R"(
def _mjpython_main():
  # Don't pollute the global namespace.
  global _mjpython_main
  del _mjpython_main

  import ctypes

  # GLFW must be initialized on the OS main thread (i.e. here).
  import glfw
  import mujoco.viewer

  glfw.init()
  glfw.poll_events()
  ctypes.CDLL(None).mjpython_hide_dock_icon()

  # Wait for Python main thread to finish setting up _MJPYTHON
  global cond
  with cond:
    cond.notify()
    cond.wait()
  del cond

  while True:
    try:
      # Wait for an incoming payload.
      task = mujoco.viewer._MJPYTHON.get()

      # None means that we are exiting.
      if task is None:
        glfw.terminate()
        break

      # Otherwise, launch the viewer.
      model, data, handle_return, key_callback, show_left_ui, show_right_ui = task
      ctypes.CDLL(None).mjpython_show_dock_icon()
      mujoco.viewer._launch_internal(
          model,
          data,
          run_physics_thread=False,
          handle_return=handle_return,
          key_callback=key_callback,
          show_left_ui=show_left_ui,
          show_right_ui=show_right_ui,
      )
      ctypes.CDLL(None).mjpython_hide_dock_icon()

    finally:
      mujoco.viewer._MJPYTHON.done()

_mjpython_main()
)", nullptr);
  cpython.PyGILState_Release(gil);

  // Tear everything down.
  pthread_join(pymain_thread, nullptr);
  dlclose(libpython);
}
