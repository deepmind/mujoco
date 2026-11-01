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

#include <mujoco/mjbatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <new>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "engine/engine_forward.h"
#include "engine/engine_io.h"
#include "engine/engine_setconst.h"
#include "engine/engine_util_errmem.h"

// setjmp on macOS saves the signal mask with a system call; _setjmp does not
#if defined(__APPLE__)
  #define mjb_setjmp _setjmp
  #define mjb_longjmp _longjmp
#else
  #define mjb_setjmp setjmp
  #define mjb_longjmp longjmp
#endif

// hint to the CPU that the thread is spin-waiting
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
static inline void CpuPause() {
  _mm_pause();
}
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
  #include <intrin.h>
static inline void CpuPause() {
  __yield();
}
#elif defined(__aarch64__) || defined(__arm__)
static inline void CpuPause() {
  __asm__ __volatile__("yield" ::: "memory");
}
#else
static inline void CpuPause() {}
#endif

namespace {


//------------------------------------------ fields -----------------------------------------------

// one field of mjModel or mjData, sized for a particular model
struct Field {
  void* (*get)(void* obj);            // pointer to the field in an mjModel or mjData
  void (*set)(void* obj, void* ptr);  // mjModel arrays: point the field at ptr
  size_t bytes;                       // bytes per simulation
  int    size;                        // elements per simulation
  int    elemsize;                    // bytes per element
  bool   state;                       // mjData: in the mjSTATE_INTEGRATION row
  bool   asset;                       // mjModel: large constant data mj_setConst never writes
  int    reads  = 0;                  // mjModel: groups of mj_setConst that read it
  int    writes = 0;                  // mjModel: groups of mj_setConst that write it
  bool   input  = false;              // mjModel: the first group that uses it reads it
};

// fields by name
using FieldTable = std::unordered_map<std::string, Field>;

// whether an mjData field is in the mjSTATE_INTEGRATION row
bool InState(std::string_view name) {
  for (std::string_view s : {"qpos",
                             "qvel",
                             "act",
                             "history",
                             "qacc_warmstart",
                             "ctrl",
                             "qfrc_applied",
                             "xfrc_applied",
                             "eq_active",
                             "mocap_pos",
                             "mocap_quat",
                             "userdata",
                             "plugin_state",
                             "plastic"}) {
    if (name == s) return true;
  }
  return false;
}


// whether models a and b give mjData the same layout
bool SameDataLayout(const mjModel* a, const mjModel* b) {
  if (a->narena != b->narena) return false;
#define X(type, name, nr, nc) \
  if (a->nr != b->nr) return false;
  MJDATA_POINTERS
#undef X
  return true;
}

// whether an mjModel field is asset data, shared by the thread models
bool IsAsset(std::string_view name) {
  for (std::string_view p : {"mesh_", "hfield_", "tex_", "skin_", "bvh_", "oct_"}) {
    if (name.substr(0, p.size()) == p) return true;
  }
  return false;
}


// field of nr x nc elements of elemsize bytes
Field MakeField(void* (*get)(void*),
                void (*set)(void*, void*),
                size_t nr,
                size_t nc,
                size_t elemsize,
                bool   state,
                bool   asset) {
  size_t size = nr * nc;
  return {get,
          set,
          size * elemsize,
          static_cast<int>(size),
          static_cast<int>(elemsize),
          state,
          asset};
}


// array fields of mjData
FieldTable DataFields(const mjModel* m) {
  FieldTable t;
#define X(type, name, nr, nc)                                                          \
  t[#name] = MakeField([](void* o) -> void* { return static_cast<mjData*>(o)->name; }, \
                       nullptr,                                                        \
                       m->nr,                                                          \
                       nc,                                                             \
                       sizeof(type),                                                   \
                       InState(#name),                                                 \
                       false);
  MJDATA_POINTERS
#undef X

  // fixed-size numeric arrays (energy, solver iteration counts, ...)
#define X(type, name, nr, nc)                                                            \
  if constexpr (std::is_arithmetic_v<type>) {                                            \
    t[#name] = MakeField([](void* o) -> void* { return static_cast<mjData*>(o)->name; }, \
                         nullptr,                                                        \
                         nr,                                                             \
                         nc,                                                             \
                         sizeof(type),                                                   \
                         false,                                                          \
                         false);                                                         \
  }
  MJDATA_VECTOR
#undef X

  // the scalars that count what a call computed (contacts, constraints, islands); the others
  // are pointers, or accumulate over the simulations that share an mjData
  static constexpr std::string_view kCounts[] =
      {"ncon", "ne", "nf", "nl", "nefc", "nJ", "nA", "nisland", "nidof"};
#define X(type, name)                                                                     \
  if (std::find(std::begin(kCounts), std::end(kCounts), #name) != std::end(kCounts)) {    \
    t[#name] = MakeField([](void* o) -> void* { return &static_cast<mjData*>(o)->name; }, \
                         nullptr,                                                         \
                         1,                                                               \
                         1,                                                               \
                         sizeof(type),                                                    \
                         false,                                                           \
                         false);                                                          \
  }
  MJDATA_SCALAR
#undef X
  return t;
}


// array fields of mjModel and fields of mjOption
FieldTable ModelFields(const mjModel* m) {
  FieldTable t;
#if defined(__clang__) || defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-variable"
#endif
  MJMODEL_POINTERS_PREAMBLE(m)
#define X(type, name, nr, nc)                                                                     \
  t[#name] =                                                                                      \
      MakeField([](void* o) -> void* { return static_cast<mjModel*>(o)->name; },                  \
                [](void* o, void* p) { static_cast<mjModel*>(o)->name = static_cast<type*>(p); }, \
                m->nr,                                                                            \
                nc,                                                                               \
                sizeof(type),                                                                     \
                false,                                                                            \
                IsAsset(#name));
  MJMODEL_POINTERS
#undef X
#if defined(__clang__) || defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

  // mjOption fields, named "opt.<name>"
#define X(type, name, n)                                                              \
  t["opt." #name] =                                                                   \
      MakeField([](void* o) -> void* { return &static_cast<mjModel*>(o)->opt.name; }, \
                nullptr,                                                              \
                1,                                                                    \
                n,                                                                    \
                sizeof(type),                                                         \
                false,                                                                \
                false);
#define XVEC(type, name, n)                                                                        \
  t["opt." #name] = MakeField([](void* o) -> void* { return static_cast<mjModel*>(o)->opt.name; }, \
                              nullptr,                                                             \
                              1,                                                                   \
                              n,                                                                   \
                              sizeof(type),                                                        \
                              false,                                                               \
                              false);
  MJOPTION_FIELDS
#undef X
#undef XVEC

  // bvh_aabb: the bodies' bounding volumes, which mj_setConst refits; the meshes' follow them
  int nbvh = 0;
  for (int i = 0; i < m->nbody; ++i) {
    if (m->body_bvhadr[i] >= 0) nbvh = std::max(nbvh, m->body_bvhadr[i] + m->body_bvhnum[i]);
  }
  Field& bvh = t["bvh_aabb"];
  bvh.size   = 6 * nbvh;
  bvh.bytes  = bvh.size * sizeof(mjtNum);
  bvh.asset  = false;

  // how the groups of mj_setConst use each array: column g of MJSETCONST_ARRAYS is group 1 << g
  static_assert(mjSCGROUP_FIXED == 1 << 0 &&
                mjSCGROUP_KIN0 == 1 << 1 &&
                mjSCGROUP_INERT0 == 1 << 2 &&
                mjSCGROUP_SPRING == 1 << 3);
#define X(name, fixed, kin0, inert0, spring)            \
  {                                                     \
    Field&    f        = t[#name];                      \
    const int access[] = {fixed, kin0, inert0, spring}; \
    for (int g = 3; g >= 0; --g) {                      \
      if (access[g] & mjSC_IN) f.reads |= 1 << g;       \
      if (access[g] & mjSC_OUT) f.writes |= 1 << g;     \
      if (access[g]) f.input = access[g] & mjSC_IN;     \
    }                                                   \
  }
  MJSETCONST_ARRAYS
#undef X

  // the options are an input of every group: mj_setConst reads the timestep and the spring,
  // damper and midphase disable flags (in setEfm0Factor and mj_flexSkip), and no other option
  // TODO(tassa): the disable flags should not change the constants, e.g. the compiler could
  // clear them while it runs mj_setConst; then only the timestep would be an input, of the
  // spring group, and the other options of no group
  for (auto& [name, f] : t) {
    if (name.substr(0, 4) == "opt.") {
      f.reads = mjSCGROUP_ALL;
      f.input = true;
    }
  }
  return t;
}


// mjModel scalars that mj_setConst writes (MJSETCONST_SCALARS); kept per simulation like
// expanded fields
struct Scalars {
#define X(name) decltype(mjModel::name) name;
  MJSETCONST_SCALARS
#undef X

  // scalars of model m
  static Scalars Of(const mjModel* m) {
    Scalars s;
#define X(name) s.name = m->name;
    MJSETCONST_SCALARS
#undef X
    return s;
  }

  // write the scalars into model m
  void Apply(mjModel* m) const {
#define X(name) m->name = name;
    MJSETCONST_SCALARS
#undef X
  }
};

// per-simulation storage for one field
struct Slot {
  const Field*               field;            // the field
  std::unique_ptr<uint8_t[]> buf;              // (nsim, bytes)
  bool                       derived = false;  // expanded because mj_setConst changed it
  std::unique_ptr<uint8_t[]> applied;  // persistent: (nsim, bytes) as of the stages in mjData

  // storage of simulation i
  uint8_t* Row(int i) { return buf.get() + static_cast<size_t>(i) * field->bytes; }

  // values of simulation i that its mjData's position and velocity stages were computed with
  uint8_t* Applied(int i) { return applied.get() + static_cast<size_t>(i) * field->bytes; }
};

// a record of mjb_rollout, resolved
struct Record {
  const Field* field;  // mjData field, or NULL for a state
  int          spec;   // mjtState bits of the state
  size_t       bytes;  // per substep
  uint8_t*     buf;    // output, (n, nstep, bytes)
};


//------------------------------------------ errors -----------------------------------------------

thread_local std::jmp_buf* tls_jmp = nullptr;  // setjmp around the running simulation's call
// message of the trapped error: the function's name, then the subject
thread_local char          tls_error[sizeof(mjLogMessage::subject) + 128];
thread_local mjfLogHandler tls_prev  = nullptr;  // the thread's handler before the batch's
thread_local const void*   tls_batch = nullptr;  // batch whose simulation the thread runs

// trap errors by unwinding to tls_jmp, pass other messages to the previous handler
void BatchLogHandler(const mjLogMessage* msg) {
  if (msg->level == mjLOG_ERROR && tls_jmp) {
    if (msg->func) {
      std::snprintf(tls_error, sizeof(tls_error), "%s: %s", msg->func, msg->subject);
    } else {
      std::snprintf(tls_error, sizeof(tls_error), "%s", msg->subject);
    }
    mjb_longjmp(*tls_jmp, 1);
  }
  (tls_prev ? tls_prev : _mjPRIVATE_getGlobalLogHandler())(msg);
}


//------------------------------------------ thread pool ------------------------------------------

// thread pool of nthread - 1 workers and the caller, which is thread nthread - 1
class Pool {
 public:
  // task run on one item
  using Fn = void (*)(void* ctx, int thread, int item);

  // start the workers; if one cannot start, join the others and rethrow
  explicit Pool(int nthread)
      : nthread_(nthread), next_(new Counter[nthread]), parked_(new bool[nthread]()) {
    try {
      workers_.reserve(nthread - 1);
      for (int t = 0; t < nthread - 1; ++t) workers_.emplace_back([this, t] { Worker(t); });
    } catch (...) {
      Stop();
      throw;
    }
  }

  // stop and join the workers
  ~Pool() { Stop(); }

  // number of threads, including the caller's
  int size() const { return nthread_; }

  // run fn(ctx, thread, item) for item in [0, n) on min(n, nthread) threads and return when
  // all are done; the other threads neither run nor wait for the call, and one item runs inline
  void Run(int n, Fn fn, void* ctx) {
    if (n == 0) return;
    int p = std::min(n, nthread_);
    fn_   = fn;
    ctx_  = ctx;
    n_    = n;
    p_    = p;
    for (int s = 0; s < p; ++s) next_[s].next.store(Start(s), std::memory_order_relaxed);
    if (p > 1) {
      active_.store(p - 1, std::memory_order_relaxed);
      bool wake = false;
      {
        std::lock_guard<std::mutex> lock(mu_);

        uint64_t call = (epoch_.load(std::memory_order_relaxed) >> 32) + 1;
        epoch_.store(call << 32 | static_cast<uint32_t>(p), std::memory_order_release);

        // wake the parked workers only if the call needs one of them: a call on fewer
        // simulations than threads leaves the others parked
        for (int t = 0; t < p - 1 && !wake; ++t) wake = parked_[t];
      }
      if (wake) wake_.notify_all();
    }
    Work(p - 1, nthread_ - 1);  // the caller takes the last slice
    if (p > 1) {
      auto done = [this] { return active_.load(std::memory_order_acquire) == 0; };
      if (!SpinUntil(done, Clock::now() + kSpin)) {
        std::unique_lock<std::mutex> lock(mu_);
        done_.wait(lock, done);
      }
    }
  }

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::microseconds kSpin{50};  // spin this long before parking

  // first item of slice s: item i starts in slice i * p / n
  int Start(int s) const { return static_cast<int>(static_cast<int64_t>(s) * n_ / p_); }

  // spin until pred holds or the deadline passes; return whether it holds
  template <typename Pred>
  static bool SpinUntil(Pred pred, Clock::time_point deadline) {
    for (int k = 0;; ++k) {
      if (pred()) return true;
      if ((k & 63) == 63 && Clock::now() > deadline) return false;
      CpuPause();
    }
  }

  // run the items of a slice, then take from the other slices
  void Work(int slice, int thread) {
    for (int k = 0; k < p_; ++k) {
      int s   = (slice + k) % p_;
      int end = Start(s + 1);
      for (int i = next_[s].next.fetch_add(1, std::memory_order_relaxed); i < end;
           i     = next_[s].next.fetch_add(1, std::memory_order_relaxed)) {
        fn_(ctx_, thread, i);
      }
    }
  }

  // worker loop: spin briefly after a call it ran, so back-to-back calls skip the wake-up,
  // then park until a call it is part of; calls it is not part of do not extend the spin
  void Worker(int thread) {
    _mjPRIVATE_setTlsLogHandler(BatchLogHandler);
    uint64_t          seen     = 0;
    Clock::time_point deadline = Clock::now() + kSpin;
    while (true) {
      auto ready = [this, &seen] {
        return stop_.load(std::memory_order_acquire) ||
               epoch_.load(std::memory_order_acquire) != seen;
      };
      if (!SpinUntil(ready, deadline)) {
        auto mine = [this, &seen, thread] {
          uint64_t epoch = epoch_.load(std::memory_order_acquire);
          return stop_.load(std::memory_order_acquire) ||
                 (epoch != seen && thread < static_cast<int>(epoch & 0xffffffff) - 1);
        };
        std::unique_lock<std::mutex> lock(mu_);
        parked_[thread] = true;
        wake_.wait(lock, mine);
        parked_[thread] = false;
      }
      if (stop_.load(std::memory_order_acquire)) return;
      seen  = epoch_.load(std::memory_order_acquire);
      int p = static_cast<int>(seen & 0xffffffff);  // participants of that call
      if (thread >= p - 1) continue;
      Work(thread, thread);
      deadline = Clock::now() + kSpin;
      if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mu_);
        done_.notify_one();
      }
    }
  }

  // stop and join the workers that were started
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_.store(true, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto& w : workers_) w.join();
  }

  // next item of a slice, on a cache line of its own
  struct alignas(64) Counter {
    std::atomic<int> next{0};
  };

  const int                  nthread_;        // threads, including the caller's
  std::vector<std::thread>   workers_;        // nthread - 1 workers
  std::unique_ptr<Counter[]> next_;           // next item of each slice
  std::unique_ptr<bool[]>    parked_;         // workers parked on wake_, guarded by mu_
  std::mutex                 mu_;             // guards parking and waking
  std::condition_variable    wake_;           // wakes parked workers for a call
  std::condition_variable    done_;           // wakes the caller when workers are done
  std::atomic<uint64_t>      epoch_{0};       // calls so far << 32 | participants of the last
  std::atomic<int>           active_{0};      // workers still running the call
  std::atomic<bool>          stop_{false};    // workers should exit
  Fn                         fn_  = nullptr;  // task of the call
  void*                      ctx_ = nullptr;  // context of the task
  int                        n_   = 0;        // number of items of the call
  int                        p_   = 1;        // threads taking part in the call
};

}  // namespace


//------------------------------------------ stages -----------------------------------------------

// compute the stages of mj_forward up to stage at the current state; up to mjSTAGE_VEL this
// is mj_step1 without the control callback, so that mj_step2 can continue from it
static void Forward(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_NONE) return;

  // the checks of mj_step1, which mj_forward lacks and a continued step skips
  if (stage == mjSTAGE_ACC) {
    mj_checkPos(m, d);
    mj_checkVel(m, d);
    mj_forward(m, d);
    return;
  }
  mj_checkDiscrete(m);
  mj_checkPos(m, d);
  mj_checkVel(m, d);
  mj_fwdPosition(m, d);
  mj_sensorPos(m, d);
  if (!d->flg_energypos) {
    if (m->opt.enableflags & mjENBL_ENERGY) {
      mj_energyPos(m, d);
    } else {
      d->energy[0] = d->energy[1] = 0;
    }
  }
  if (stage == mjSTAGE_VEL) {
    mj_fwdVelocity(m, d);
    mj_sensorVel(m, d);
    if ((m->opt.enableflags & mjENBL_ENERGY) && !d->flg_energyvel) mj_energyVel(m, d);
  }
}


//------------------------------------------ batch ------------------------------------------------

// batch of simulations of one model
struct mjBatch_ {
  // operation of a call; Name() lists their API functions in this order
  enum class Op { Step, Forward, Reset, SetConst, Rollout, Apply };

  int                                nsim;             // number of simulations
  int                                nstate;           // size of a state row
  bool                               persistent;       // one mjData per simulation
  mjModel*                           model = nullptr;  // the batch's copy
  FieldTable                         data_fields;      // mjData fields, by name
  FieldTable                         model_fields;     // mjModel and mjOption fields, by name
  std::vector<const Field*>          restorable;       // non-asset model fields
  std::vector<const Field*>          setconst_out;     // model fields mj_setConst writes
  std::vector<mjtNum>                state;            // (nsim, nstate)
  std::vector<mjWarningStat>         warning;          // (nsim, mjNWARNING)
  std::vector<Scalars>               scalars;          // per simulation
  std::vector<int>                   status;           // per simulation
  std::vector<std::string>           errors;           // per simulation
  std::vector<std::unique_ptr<Slot>> outputs;          // registered mjData fields
  std::vector<std::unique_ptr<Slot>> expanded;         // per-simulation mjModel fields
  std::set<const Field*>             expanded_set;     // fields of expanded
  std::vector<mjData*>               data;           // per thread, or per simulation if persistent
  std::vector<mjData*>               scratch;        // per thread, for set_const if persistent
  std::vector<uint8_t>               current;        // persistent: mjData has the stages of the row
  std::vector<std::vector<mjtNum>>   loaded;         // persistent: per thread, a state of an mjData
  int                                input_adr = 0;  // ctrl, qfrc_applied, xfrc_applied in a row
  int                                input_end = 0;  // their end
  std::vector<mjModel*>              models;         // per thread, once a field is expanded
  std::vector<const mjModel*>        ext;  // per simulation: the caller's models, or empty
  std::vector<void*>                 model_buffers;  // the arrays they own
  std::vector<mjtNum*>               bvh_bodies;     // per thread: its bodies' bounding volumes
  std::vector<mjtNum*>               bvh_all;        // per thread: all bounding volumes
  std::set<const Field*>             changed;        // set_const outputs not yet expanded
  std::unique_ptr<Pool>              pool;           // the batch's threads
  std::mutex                         mu;             // serializes calls
  std::mutex                         changed_mu;     // changed, from threads

  // call arguments
  Op                  op;            // operation
  int                 arg;           // nstep
  const int*          keys;          // reset: keyframe per position in the call, or NULL
  int                 stage;         // step: the stages computed at the new state
  int                 groups;        // set_const: the groups of mj_setConst to run
  const int*          ids;           // simulations, NULL: all
  mjfBatchFunc        func;          // apply: function
  void*               func_arg;      // apply: its argument
  int                 save;          // apply: save the state
  const mjtNum*       control;       // rollout: controls, or NULL
  int                 control_spec;  // rollout: state components of the controls
  int                 ncontrol;      // rollout: size of a control
  std::vector<Record> records;       // rollout: records

  // state row of simulation i
  mjtNum* State(int i) { return state.data() + static_cast<size_t>(i) * nstate; }

  // warning counters of simulation i
  mjWarningStat* Warning(int i) { return warning.data() + static_cast<size_t>(i) * mjNWARNING; }

  // restore the fields and scalars mj_setConst writes, and the options, of a thread's model; the
  // options because an error in mj_setConst leaves the sleep flag cleared
  void Restore(mjModel* m) {
    for (const Field* f : setconst_out) std::memcpy(f->get(m), f->get(model), f->bytes);
    Scalars::Of(model).Apply(m);
    m->opt = model->opt;
  }

  // whether the per-simulation values of s are an input of mj_setConst: the field is not derived,
  // and the first group that uses it reads it rather than writing it
  static bool SetConstInput(const Slot& s) { return !s.derived && s.field->input; }

  // the groups of mj_setConst to run: those that read a per-simulation input, and those that read
  // what an earlier one of them writes; the others give every simulation the model's constants,
  // since their inputs are the model's
  int SetConstGroups() const {
    std::set<const Field*> affected;  // per-simulation inputs, and outputs of groups to run
    for (auto& s : expanded) {
      if (SetConstInput(*s)) affected.insert(s->field);
    }
    int run = 0;
    for (int g = 0; g < 4; ++g) {
      int  group = 1 << g;
      bool reads = false;
      for (const Field* f : affected) reads = reads || (f->reads & group);
      if (!reads) continue;
      run |= group;
      for (const Field* f : setconst_out) {
        if (f->writes & group) affected.insert(f);
      }
    }
    return run;
  }

  // give model field f per-simulation storage, seeded from the model
  Slot& Expand(const Field& f) {
    auto s   = std::make_unique<Slot>();
    s->field = &f;
    s->buf   = std::make_unique<uint8_t[]>(f.bytes * nsim);
    for (int i = 0; i < nsim; ++i) std::memcpy(s->Row(i), f.get(model), f.bytes);
    if (persistent) {  // the model's values, which the stages in mjData were computed with
      s->applied = std::make_unique<uint8_t[]>(f.bytes * nsim);
      std::memcpy(s->applied.get(), s->buf.get(), f.bytes * nsim);
    }
    expanded.push_back(std::move(s));
    expanded_set.insert(&f);
    if (models.empty()) {
      for (int t = 0; t < pool->size(); ++t) {
        models.push_back(ShallowCopy());
        bvh_bodies.push_back(models.back()->bvh_aabb);
        bvh_all.push_back(model->bvh_aabb);
      }
    } else if (f.set && !f.writes) {
      // the thread models shared the field, and now write each simulation's values into it
      for (mjModel* m : models) Own(m, f);
    }

    // per-simulation bounding volumes: each thread model gets the whole array
    if (&f == &model_fields.at("bvh_aabb")) {
      size_t bytes = sizeof(mjtNum) * 6 * model->nbvhstatic;
      for (mjtNum*& bvh : bvh_all) {
        void* buf = ::operator new(bytes ? bytes : 1, std::align_val_t(64));
        model_buffers.push_back(buf);
        std::memcpy(buf, model->bvh_aabb, bytes);
        bvh = static_cast<mjtNum*>(buf);
      }
    }
    return *expanded.back();
  }

  // whether a thread model writes field f, and so needs its own copy: mj_setConst writes it, or
  // each simulation's values are written into it
  bool Writes(const Field& f) const { return f.set && (f.writes || expanded_set.count(&f)); }

  // model for one thread: a copy of the struct with its own copies of the arrays it writes,
  // sharing the others with the batch's model, which nothing writes
  mjModel* ShallowCopy() {
    constexpr size_t kAlign = 64;
    size_t           total  = 0;
    for (auto& [name, f] : model_fields) {
      if (Writes(f)) total += (f.bytes + kAlign - 1) / kAlign * kAlign;
    }
    auto* buf =
        static_cast<uint8_t*>(::operator new(total ? total : kAlign, std::align_val_t(kAlign)));
    model_buffers.push_back(buf);
    auto*  m      = new mjModel(*model);
    size_t offset = 0;
    for (auto& [name, f] : model_fields) {
      if (!Writes(f)) continue;
      std::memcpy(buf + offset, f.get(model), f.bytes);
      f.set(m, buf + offset);
      offset += (f.bytes + kAlign - 1) / kAlign * kAlign;
    }
    m->buffer  = nullptr;  // not an mjModel to free with mj_deleteModel
    m->nbuffer = 0;
    return m;
  }

  // give thread model m its own copy of model field f, which it shared
  void Own(mjModel* m, const Field& f) {
    void* buf = ::operator new(f.bytes ? f.bytes : 1, std::align_val_t(64));
    model_buffers.push_back(buf);
    std::memcpy(buf, f.get(model), f.bytes);
    f.set(m, buf);
  }

  // number of warnings raised in d
  int WarningCount(const mjData* d) {
    int n = 0;
    for (int w = 0; w < mjNWARNING; ++w) n += d->warning[w].number;
    return n;
  }

  // copy record r of substep k for call position j out of d, or repeat substep k - 1
  void CopyRecord(const Record& r, const mjModel* m, mjData* d, int j, int k, bool repeat) {
    uint8_t* dst = r.buf + (static_cast<size_t>(j) * arg + k) * r.bytes;
    if (repeat) {
      std::memcpy(dst, dst - r.bytes, r.bytes);
    } else if (r.field) {
      std::memcpy(dst, r.field->get(d), r.bytes);
    } else {
      mj_getState(m, d, reinterpret_cast<mjtNum*>(dst), r.spec);
    }
  }

  // run the current op on simulation i, at position j of the call, on thread t; may longjmp
  void RunSim(int t, int i, int j) {
    // the batch's model, or the thread's copy with per-simulation fields; with caller models,
    // which exclude per-simulation fields, simulation i's own
    mjModel*       own = models.empty() ? model : models[t];
    const mjModel* m   = ext.empty() ? own : ext[i];
    mjData*        d   = Data(t, i);
    if (!models.empty()) {
      // mj_setConst writes only the bodies' bounding volumes; the pipeline reads the meshes' too
      own->bvh_aabb = op == Op::SetConst ? bvh_bodies[t] : bvh_all[t];
    }
    if (op == Op::SetConst) Restore(own);
    for (auto& s : expanded) {
      // mj_setConst recomputes its outputs from the template's, as on a fresh copy
      if (op == Op::SetConst && s->field->writes && !SetConstInput(*s)) continue;
      std::memcpy(s->field->get(own), s->Row(i), s->field->bytes);
    }

    if (op == Op::SetConst) {
      mj_setConstGroups(own, d, groups);
      for (auto& s : expanded) std::memcpy(s->Row(i), s->field->get(own), s->field->bytes);
      scalars[i] = Scalars::Of(own);
      for (const Field* f : setconst_out) {
        if ((f->writes & groups) &&
            !expanded_set.count(f) &&
            std::memcmp(f->get(own), f->get(model), f->bytes)) {
          std::lock_guard<std::mutex> lock(changed_mu);
          changed.insert(f);
        }
      }
      Restore(own);
      if (persistent) current[i] = 0;  // the model's constants changed
      return;
    }

    if (!models.empty()) scalars[i].Apply(own);
    if (!persistent && (m->opt.enableflags & (mjENBL_SLEEP | mjENBL_IPC))) {
      mju_error("expanded opt.enableflags enables sleep or IPC, which require a persistent batch");
    }
    bool continues = op == Op::Step && Continues(t, i, m, d);
    if (op == Op::Reset) {
      int key = keys ? keys[j] : -1;
      if (key >= 0) {
        mj_resetDataKeyframe(m, d, key);
      } else {
        mj_resetData(m, d);
      }
    } else {
      mj_setState(m, d, State(i), mjSTATE_INTEGRATION);
      std::memcpy(d->warning, Warning(i), sizeof(d->warning));
    }

    switch (op) {
      case Op::Step: {
        // mj_step1 + mj_step2 is mj_step: continue from the stages the last call left
        int k = 0;

        // mj_step2 uncounts the step of mj_step1, which Forward did not count
        if (continues) {
          d->timer[mjTIMER_STEP].number++;
          mj_step2(m, d);
          k = 1;
        }
        for (; k < arg; ++k) mj_step(m, d);
        Forward(m, d, stage);
        break;
      }
      case Op::Forward:
      case Op::Reset:
        Forward(m, d, stage);
        break;
      case Op::Rollout: {
        int  warnings = WarningCount(d);
        bool stopped  = false;
        for (int k = 0; k < arg; ++k) {
          if (!stopped) {
            if (control) {
              size_t row = static_cast<size_t>(j) * arg + k;
              mj_setState(m, d, control + row * ncontrol, control_spec);
            }
            mj_step(m, d);
          }
          for (const Record& r : records) CopyRecord(r, m, d, j, k, stopped);
          stopped = stopped || WarningCount(d) > warnings;
        }
        break;
      }
      case Op::Apply:
        func(m, d, i, func_arg);
        break;
      case Op::SetConst:
        break;
    }

    if (op != Op::Apply || save) {
      mj_getState(m, d, State(i), mjSTATE_INTEGRATION);
      std::memcpy(Warning(i), d->warning, sizeof(d->warning));
    }

    // apply's function may compute any subset of the fields, so apply publishes none
    if (op != Op::Apply) {
      for (auto& s : outputs) std::memcpy(s->Row(i), s->field->get(d), s->field->bytes);
    }

    // a persistent mjData whose position and velocity stages match its row can continue
    if (persistent) {
      current[i] = (op == Op::Step || op == Op::Forward || op == Op::Reset) && stage >= mjSTAGE_VEL;
      if (current[i]) {
        for (auto& s : expanded) std::memcpy(s->Applied(i), s->Row(i), s->field->bytes);
      }
    }
  }

  // whether simulation i's next step can start with mj_step2: its mjData holds the position
  // and velocity stages of the state and model fields it is about to load, apart from the
  // inputs mj_step2 reads, and nothing outside them can differ
  bool Continues(int t, int i, const mjModel* m, const mjData* d) {
    if (!persistent || !current[i]) return false;
    if (m->opt.integrator == mjINT_RK4) return false;  // mj_step2 integrates with Euler
    if (mjcb_control || mjcb_passive || mjcb_contactfilter || mjcb_sensor) return false;
    mjtNum* loaded_row = loaded[t].data();
    mj_getState(m, d, loaded_row, mjSTATE_INTEGRATION);
    const mjtNum* row = State(i);
    // with a tree asleep only ctrl may differ: applied forces wake trees in mj_wake, which runs
    // in the position stage
    bool asleep = (m->opt.enableflags & mjENBL_SLEEP) && d->ntree_awake < m->ntree;
    int  end    = asleep ? input_adr + m->nu : input_end;
    if (std::memcmp(loaded_row, row, sizeof(mjtNum) * input_adr) ||
        std::memcmp(loaded_row + end, row + end, sizeof(mjtNum) * (nstate - end))) {
      return false;
    }
    for (auto& s : expanded) {
      if (std::memcmp(s->Row(i), s->Applied(i), s->field->bytes)) return false;
    }
    return true;
  }

  // mjData for simulation i on thread t: set_const in a persistent batch uses scratch
  // data, since mj_setConst would clobber the simulation's sleep state and be misled by it
  mjData* Data(int t, int i) {
    if (!persistent) return data[t];
    return op == Op::SetConst ? scratch[t] : data[i];
  }

  // run simulation i, trapping a MuJoCo error into its status
  void Guarded(int t, int i, int j) {
    std::jmp_buf  jb;
    std::jmp_buf* outer_jmp   = tls_jmp;  // an enclosing batch call's, when nested
    const void*   outer_batch = tls_batch;

    tls_jmp   = &jb;
    tls_batch = this;
    status[i] = 0;
    if (mjb_setjmp(jb) == 0) {
      RunSim(t, i, j);
      errors[i].clear();
    } else {
      // the simulation's state was not saved; its mjData is reset for the next call.
      // mj_resetData is assumed infallible (warns rather than calling mju_error): trapping a
      // longjmp mid-reset would leak plugin state, overwrite tls_error, and leave a half-reset
      // mjData (open stack frames, poisoned arena) for the next simulation sharing Data(t, i)
      tls_jmp = outer_jmp;
      if (op == Op::SetConst) Restore(models[t]);

      // the simulation's model, or the template after set_const, whose thread model then holds
      // only the bodies' bounding volumes
      mj_resetData(models.empty() || op == Op::SetConst ? model : models[t], Data(t, i));
      if (persistent) current[i] = 0;
      status[i] = 1;
      errors[i] = tls_error;
    }
    tls_jmp   = outer_jmp;
    tls_batch = outer_batch;
  }

  // pool task: run the simulation at position j of the call
  static void Task(void* ctx, int thread, int j) {
    auto* b = static_cast<mjBatch_*>(ctx);
    b->Guarded(thread, b->ids ? b->ids[j] : j, j);
  }

  // run fn on n items on the pool, with the caller's errors trapped like the workers'
  void RunTrapped(int n, Pool::Fn fn, void* ctx) {
    mjfLogHandler handler = _mjPRIVATE_setTlsLogHandler(BatchLogHandler);
    mjfLogHandler prev    = tls_prev;
    if (handler != BatchLogHandler) tls_prev = handler;  // not nested in a batch call
    pool->Run(n, fn, ctx);
    _mjPRIVATE_setTlsLogHandler(handler);
    tls_prev = prev;
  }

  // run on the selection
  void RunLocked(const int* sel, int n) {
    ids = sel;
    RunTrapped(n, &Task, this);
  }

  // number of failed simulations in the selection
  int Failures(const int* sel, int n) {
    int nfail = 0;
    for (int j = 0; j < n; ++j) nfail += status[sel ? sel[j] : j] != 0;
    return nfail;
  }

  // run op o with argument a on the selection; return the number of failed simulations
  int Run(Op o, const int* sel, int nid, int a, int s = mjSTAGE_NONE, const int* key = nullptr) {
    // validate before locking: mju_error may not return, and must not leave mu held
    CheckIds(Name(o), sel, nid);
    int n = sel ? nid : nsim;

    std::lock_guard<std::mutex> lock(mu);
    op    = o;
    arg   = a;
    stage = s;
    keys  = key;

    if (o != Op::SetConst) {
      RunLocked(sel, n);
      return Failures(sel, n);
    }

    // Expand the fields mj_setConst changed and recompute the call's simulations, until no
    // field changes outside the expanded ones. The others need nothing: an expanded field
    // starts from the model's value, which is theirs as long as their own set_const did not
    // change it. A simulation that raises keeps its constants, since nothing is written back
    // for it, and raises again in each round.
    groups = SetConstGroups();
    if (!groups) {
      // no group reads a per-simulation field: every simulation has the model's constants,
      // including in the expanded fields that mj_setConst writes without reading
      for (int j = 0; j < n; ++j) {
        int i     = sel ? sel[j] : j;
        status[i] = 0;
        errors[i].clear();
        for (auto& slot : expanded) {
          if (slot->field->writes && !SetConstInput(*slot)) {
            std::memcpy(slot->Row(i), slot->field->get(model), slot->field->bytes);
          }
        }
      }
      return 0;
    }
    while (true) {
      RunLocked(sel, n);
      if (changed.empty()) break;
      for (const Field* f : changed) Expand(*f).derived = true;
      changed.clear();
    }
    return Failures(sel, n);
  }

  // make the mjData of a list on the pool, each on the thread that will usually run it, so
  // that its pages are near that thread; per-thread lists put item t on thread t. Return the
  // first error raised while making one; the list owns every mjData that was allocated
  std::string MakeData(std::vector<mjData*>& list) {
    struct Args {
      const mjModel*        model;
      std::vector<mjData*>* list;
      std::string           error;
      std::mutex            mu;
    } args{model, &list, {}, {}};
    auto make = +[](void* ctx, int thread, int item) {
      auto*         a = static_cast<Args*>(ctx);
      std::jmp_buf  jb;
      std::jmp_buf* outer_jmp = tls_jmp;
      tls_jmp                 = &jb;
      if (mjb_setjmp(jb) == 0) {
        // mj_makeData, with the reset, which runs mj_forward when trees start asleep, after
        // the list owns the data
        mjData* d = nullptr;
        mj_makeRawData(&d, a->model);
        if (d) {
          mj_initPlugin(a->model, d);  // frees d if a plugin fails
          (*a->list)[item] = d;
          mj_resetData(a->model, d);
        }
      } else {
        // a reset abandoned mid-call leaves stack frames, which mj_deleteData would report
        if (mjData* d = (*a->list)[item]) d->pstack = d->pbase = 0;
        std::lock_guard<std::mutex> lock(a->mu);
        if (a->error.empty()) a->error = tls_error;
      }
      tls_jmp = outer_jmp;
    };
    RunTrapped(static_cast<int>(list.size()), make, &args);
    return args.error;
  }

  // name of the API function that runs op o, for error messages
  static const char* Name(Op o) {
    // in the order of Op
    static constexpr const char* kNames[] =
        {"mjb_step", "mjb_forward", "mjb_reset", "mjb_setConst", "mjb_rollout", "mjb_apply"};
    return kNames[static_cast<int>(o)];
  }

  // check that API function api is not called from this batch's own simulation (it would
  // deadlock), and that the selection is sorted, unique and in range
  void CheckIds(const char* api, const int* sel, int nid) {
    CheckNotNested(api);
    if (!sel) return;
    if (nid < 0) mju_error("%s: nid must be non-negative, got %d", api, nid);
    for (int j = 0; j < nid; ++j) {
      if (sel[j] < 0 || sel[j] >= nsim || (j > 0 && sel[j] <= sel[j - 1])) {
        mju_error("%s: simulation ids must be sorted, unique and in [0, %d); ids[%d] is %d",
                  api,
                  nsim,
                  j,
                  sel[j]);
      }
    }
  }

  // whether the simulations run on caller models
  bool CallerModels() {
    std::lock_guard<std::mutex> lock(mu);
    return !ext.empty();
  }

  // the model simulation i runs on, but for its per-simulation fields
  const mjModel* SimModel(int i) {
    std::lock_guard<std::mutex> lock(mu);
    return ext.empty() ? model : ext[i];
  }

  // raise if API function api is called from a function running on one of this batch's
  // simulations
  void CheckNotNested(const char* api) {
    if (tls_batch == this) {
      mju_error("%s: a function run by mjb_apply called its own batch", api);
    }
  }
};


//------------------------------------------ C API ------------------------------------------------

// set default batch parameters
void mjb_defaultConfig(mjBatchConfig* config) {
  config->nthread    = 0;
  config->persistent = 0;
}


// allocate a batch of nsim simulations of a copy of m
mjBatch* mjb_makeBatch(
    const mjModel* m, int nsim, const mjBatchConfig* config, char* error, int error_sz) {
  auto fail = [&](const char* msg) -> mjBatch* {
    if (error && error_sz > 0) std::snprintf(error, error_sz, "%s", msg);
    return nullptr;
  };
  mjBatchConfig defaults;
  mjb_defaultConfig(&defaults);
  if (!config) config = &defaults;
  int  nthread    = config->nthread;
  bool persistent = config->persistent != 0;
  if (nsim < 1) return fail("nsim must be positive");
  if (!persistent && (m->opt.enableflags & (mjENBL_SLEEP | mjENBL_IPC))) {
    return fail(
        "sleep and IPC require a persistent batch: their bookkeeping is not part of the "
        "state");
  }
  if (error && error_sz > 0) error[0] = '\0';

  // allocation failures, including a thread that cannot start, free what was made
  mjBatch_* b = nullptr;
  try {
    b             = new mjBatch_;
    b->nsim       = nsim;
    b->persistent = persistent;
    b->model      = mj_copyModel(nullptr, m);
    if (!b->model) {
      mjb_deleteBatch(b);
      return fail("could not copy the model");
    }
    b->nstate       = mj_stateSize(b->model, mjSTATE_INTEGRATION);
    b->data_fields  = DataFields(b->model);
    b->model_fields = ModelFields(b->model);
    for (auto& [name, f] : b->model_fields) {
      if (!f.asset) b->restorable.push_back(&f);
      if (f.writes) b->setconst_out.push_back(&f);
    }
    b->scalars.assign(nsim, Scalars::Of(b->model));
    b->status.assign(nsim, 0);
    b->errors.resize(nsim);

    if (nthread <= 0) {
      nthread = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    b->pool          = std::make_unique<Pool>(std::min(nthread, nsim));
    int nthread_pool = b->pool->size();
    b->data.assign(b->persistent ? nsim : nthread_pool, nullptr);
    b->scratch.assign(b->persistent ? nthread_pool : 0, nullptr);
    std::string make_error = b->MakeData(b->data);
    if (make_error.empty()) make_error = b->MakeData(b->scratch);
    if (!make_error.empty()) {
      mjb_deleteBatch(b);
      return fail(make_error.c_str());
    }
    for (const auto* list : {&b->data, &b->scratch}) {
      for (mjData* d : *list) {
        if (!d) {
          mjb_deleteBatch(b);
          return fail("could not allocate mjData");
        }
      }
    }

    b->state.resize(static_cast<size_t>(nsim) * b->nstate);
    b->warning.resize(static_cast<size_t>(nsim) * mjNWARNING);
    b->input_adr = mj_stateSize(b->model, mjSTATE_INTEGRATION & (mjSTATE_CTRL - 1));
    b->input_end =
        b->input_adr +
        mj_stateSize(b->model, mjSTATE_CTRL | mjSTATE_QFRC_APPLIED | mjSTATE_XFRC_APPLIED);
    if (b->persistent) {
      b->current.assign(nsim, 0);
      b->loaded.assign(nthread_pool, std::vector<mjtNum>(b->nstate));
    }
  } catch (const std::exception& e) {
    mjb_deleteBatch(b);
    return fail(e.what());
  }

  for (int i = 0; i < nsim; ++i) {
    mj_getState(b->model, b->data[0], b->State(i), mjSTATE_INTEGRATION);
  }
  return b;
}


// free a batch
void mjb_deleteBatch(mjBatch* b) {
  if (!b) return;
  if (tls_batch == b) mju_error("mjb_deleteBatch: a function run by mjb_apply deleted its batch");
  b->pool.reset();
  for (mjData* d : b->data) mj_deleteData(d);
  for (mjData* d : b->scratch) mj_deleteData(d);
  for (mjModel* m : b->models) delete m;
  for (void* buf : b->model_buffers) ::operator delete(buf, std::align_val_t(64));
  mj_deleteModel(b->model);
  delete b;
}


// number of simulations
int mjb_nsim(const mjBatch* b) {
  return b->nsim;
}


// number of threads, including the caller's
int mjb_nthread(const mjBatch* b) {
  return b->pool->size();
}


// whether the batch keeps one mjData per simulation
int mjb_persistent(const mjBatch* b) {
  return b->persistent;
}


// the batch's copy of the model
const mjModel* mjb_model(const mjBatch* b) {
  return b->model;
}


// size of a state row
int mjb_nstate(const mjBatch* b) {
  return b->nstate;
}


// state rows
mjtNum* mjb_state(mjBatch* b) {
  return b->state.data();
}


// address of a state component in the first row
mjtNum* mjb_stateComponent(mjBatch* b, int component, int* size) {
  if (component <= 0 || component & (component - 1) || !(component & mjSTATE_INTEGRATION)) {
    mju_error("mjb_stateComponent: component must be one mjtState component");
  }
  if (size) *size = mj_stateSize(b->model, component);
  return b->state.data() + mj_stateSize(b->model, mjSTATE_INTEGRATION & (component - 1));
}


// warning counters
mjWarningStat* mjb_warning(mjBatch* b) {
  return b->warning.data();
}


// register an mjData field to copy out after every call, return its storage
void* mjb_output(mjBatch* b, const char* name, int* size, int* elemsize) {
  b->CheckNotNested("mjb_output");
  std::lock_guard<std::mutex> lock(b->mu);
  auto                        it = b->data_fields.find(name);
  if (it == b->data_fields.end() || it->second.state) return nullptr;
  const Field& f = it->second;
  if (size) *size = f.size;
  if (elemsize) *elemsize = f.elemsize;
  for (auto& s : b->outputs) {
    if (s->field == &f) return s->buf.get();
  }
  auto s   = std::make_unique<Slot>();
  s->field = &f;
  s->buf   = std::make_unique<uint8_t[]>(f.bytes * b->nsim);  // zero until a call fills it
  if (b->persistent) {  // each simulation's mjData has its current values
    for (int i = 0; i < b->nsim; ++i) std::memcpy(s->Row(i), f.get(b->data[i]), f.bytes);
  }
  b->outputs.push_back(std::move(s));
  return b->outputs.back()->buf.get();
}


// give an mjModel field per-simulation values, return their storage
void* mjb_expand(mjBatch* b, const char* name, int* size, int* elemsize) {
  b->CheckNotNested("mjb_expand");
  if (b->CallerModels()) mju_error("mjb_expand: the batch runs on caller models");
  std::lock_guard<std::mutex> lock(b->mu);
  auto                        it = b->model_fields.find(name);
  if (it == b->model_fields.end() || it->second.asset) return nullptr;
  const Field& f = it->second;
  if (size) *size = f.size;
  if (elemsize) *elemsize = f.elemsize;
  for (auto& s : b->expanded) {
    if (s->field != &f) continue;
    s->derived = false;  // expanded explicitly: an input of mj_setConst from now on
    return s->buf.get();
  }
  return b->Expand(f).buf.get();
}


// per-simulation values of an mjModel field if it is expanded, changing nothing
const void* mjb_expanded(mjBatch* b, const char* name, int* size, int* elemsize) {
  b->CheckNotNested("mjb_expanded");
  std::lock_guard<std::mutex> lock(b->mu);
  auto                        it = b->model_fields.find(name);
  if (it == b->model_fields.end() || it->second.asset) return nullptr;
  const Field& f = it->second;
  if (size) *size = f.size;
  if (elemsize) *elemsize = f.elemsize;
  for (auto& s : b->expanded) {
    if (s->field == &f) return s->buf.get();
  }
  return nullptr;
}


// advance simulations by nstep calls to mj_step
int mjb_step(mjBatch* b, const int* ids, int nid, int nstep, int stage) {
  if (nstep < 1) mju_error("mjb_step: nstep must be positive");
  if (stage < mjSTAGE_NONE || stage > mjSTAGE_ACC) {
    mju_error("mjb_step: stage must be in [mjSTAGE_NONE, mjSTAGE_ACC] = [0, %d], got %d",
              mjSTAGE_ACC,
              stage);
  }
  return b->Run(mjBatch_::Op::Step, ids, nid, nstep, stage);
}


// run mj_forward on simulations
int mjb_forward(mjBatch* b, const int* ids, int nid, int stage) {
  if (stage < mjSTAGE_NONE || stage > mjSTAGE_ACC) {
    mju_error("mjb_forward: stage must be in [mjSTAGE_NONE, mjSTAGE_ACC] = [0, %d], got %d",
              mjSTAGE_ACC,
              stage);
  }
  return b->Run(mjBatch_::Op::Forward, ids, nid, 0, stage);
}


// reset simulations to defaults or a keyframe, then run mj_forward
int mjb_reset(mjBatch* b, const int* ids, int nid, const int* key, int stage) {
  if (stage < mjSTAGE_NONE || stage > mjSTAGE_ACC) {
    mju_error("mjb_reset: stage must be in [mjSTAGE_NONE, mjSTAGE_ACC] = [0, %d], got %d",
              mjSTAGE_ACC,
              stage);
  }
  b->CheckIds("mjb_reset", ids, nid);
  int n = ids ? nid : b->nsim;
  for (int j = 0; key && j < n; ++j) {
    // keyframes are those of each simulation's model
    int nkey = static_cast<int>(b->SimModel(ids ? ids[j] : j)->nkey);
    if (key[j] < -1 || key[j] >= nkey) {
      mju_error("mjb_reset: keyframes must be in [-1, %d), got %d", nkey, key[j]);
    }
  }
  return b->Run(mjBatch_::Op::Reset, ids, nid, 0, stage, key);
}


// simulation sim's model: the batch's with its per-simulation fields and constants
void mjb_getModel(mjBatch* b, int sim, mjModel* m) {
  if (sim < 0 || sim >= b->nsim) {
    mju_error("mjb_getModel: sim must be in [0, %d), got %d", b->nsim, sim);
  }
  b->CheckNotNested("mjb_getModel");
  const mjModel* source = b->SimModel(sim);
  bool           same   = m->nbuffer == source->nbuffer;
#define X(name) same = same && m->name == source->name;
  MJMODEL_SIZES
#undef X
  if (!same) mju_error("mjb_getModel: model sizes differ from the simulation's");
  std::lock_guard<std::mutex> lock(b->mu);
  if (source != b->model) {
    mj_copyModel(m, source);
    return;
  }
  for (const Field* f : b->restorable) std::memcpy(f->get(m), f->get(b->model), f->bytes);
  for (auto& s : b->expanded) std::memcpy(s->field->get(m), s->Row(sim), s->field->bytes);
  b->scalars[sim].Apply(m);
}


// simulation sim's state, and in a persistent batch its whole mjData
void mjb_getData(mjBatch* b, int sim, mjData* d) {
  if (sim < 0 || sim >= b->nsim) {
    mju_error("mjb_getData: sim must be in [0, %d), got %d", b->nsim, sim);
  }
  if (d->nbuffer != b->data[0]->nbuffer || d->narena != b->data[0]->narena) {
    mju_error("mjb_getData: data sizes differ from the batch's");
  }
  b->CheckNotNested("mjb_getData");
  std::lock_guard<std::mutex> lock(b->mu);
  if (b->persistent) mj_copyData(d, b->model, b->data[sim]);
  mj_setState(b->model, d, b->State(sim), mjSTATE_INTEGRATION);
  std::memcpy(d->warning, b->Warning(sim), sizeof(d->warning));
}


// run mj_setConst on each simulation's model
int mjb_setConst(mjBatch* b, const int* ids, int nid) {
  b->CheckNotNested("mjb_setConst");
  if (b->CallerModels()) mju_error("mjb_setConst: the batch runs on caller models");
  return b->Run(mjBatch_::Op::SetConst, ids, nid, 0);
}


// run each simulation on a model of the caller's, or on the batch's model
void mjb_setModels(mjBatch* b, const mjModel* const* models) {
  b->CheckNotNested("mjb_setModels");
  for (int i = 0; models && i < b->nsim; ++i) {
    const mjModel* m = models[i];
    if (i > 0 && m == models[i - 1]) continue;  // checked already
    if (!m) mju_error("mjb_setModels: model %d is NULL", i);
    if (!SameDataLayout(m, b->model)) {
      mju_error("mjb_setModels: model %d does not give mjData the batch model's layout", i);
    }
    if (!b->persistent && (m->opt.enableflags & (mjENBL_SLEEP | mjENBL_IPC))) {
      mju_error("mjb_setModels: model %d enables sleep or IPC, which require a persistent batch",
                i);
    }
  }
  bool expanded;
  {
    std::lock_guard<std::mutex> lock(b->mu);
    expanded = models && !b->expanded.empty();
    if (!expanded) {
      if (models) {
        b->ext.assign(models, models + b->nsim);
      } else {
        b->ext.clear();
      }
      // the stages in a persistent batch's mjData were computed with the previous models
      std::fill(b->current.begin(), b->current.end(), 0);
    }
  }
  // raised outside the lock: mju_error may not return
  if (expanded) mju_error("mjb_setModels: the batch has per-simulation model fields");
}


// advance simulations with per-substep controls and records
int mjb_rollout(mjBatch*             b,
                const int*           ids,
                int                  nid,
                int                  nstep,
                const mjtNum*        control,
                int                  control_spec,
                const mjBatchRecord* record,
                int                  nrecord) {
  if (nstep < 1) mju_error("mjb_rollout: nstep must be positive");
  if (control && (control_spec & ~mjSTATE_USER)) {
    mju_error("mjb_rollout: control_spec must be a subset of mjSTATE_USER");
  }
  // validate everything before allocating: mju_error may not return
  if (nrecord > 0 && !record) mju_error("mjb_rollout: record is NULL but nrecord is positive");
  for (int r = 0; r < nrecord; ++r) {
    const mjBatchRecord& rec = record[r];
    if (!rec.buf) mju_error("mjb_rollout: record %d has no buffer", r);
    if (rec.field && !b->data_fields.count(rec.field)) {
      mju_error("mjb_rollout: unknown field '%s'", rec.field);
    }
    if (!rec.field && (rec.spec < 0 || rec.spec >= (1 << mjNSTATE))) {
      mju_error("mjb_rollout: invalid state spec %d in record %d", rec.spec, r);
    }
  }
  b->CheckIds("mjb_rollout", ids, nid);

  std::lock_guard<std::mutex> lock(b->mu);
  b->control      = control;
  b->control_spec = control_spec;
  b->ncontrol     = control ? mj_stateSize(b->model, control_spec) : 0;
  b->records.clear();
  for (int r = 0; r < nrecord; ++r) {
    const mjBatchRecord& rec = record[r];
    auto*                buf = static_cast<uint8_t*>(rec.buf);
    if (rec.field) {
      const Field& f = b->data_fields.at(rec.field);
      b->records.push_back({&f, 0, f.bytes, buf});
    } else {
      size_t bytes = sizeof(mjtNum) * mj_stateSize(b->model, rec.spec);
      b->records.push_back({nullptr, rec.spec, bytes, buf});
    }
  }
  b->op  = mjBatch_::Op::Rollout;
  b->arg = nstep;
  int n  = ids ? nid : b->nsim;
  b->RunLocked(ids, n);
  b->records.clear();
  return b->Failures(ids, n);
}


// run func on each simulation
int mjb_apply(mjBatch* b, const int* ids, int nid, mjfBatchFunc func, void* arg, int save) {
  if (!func) mju_error("mjb_apply: func is NULL");
  b->CheckIds("mjb_apply", ids, nid);
  std::lock_guard<std::mutex> lock(b->mu);
  b->func     = func;
  b->func_arg = arg;
  b->save     = save;
  b->op       = mjBatch_::Op::Apply;
  b->arg      = 0;
  int n       = ids ? nid : b->nsim;
  b->RunLocked(ids, n);
  return b->Failures(ids, n);
}


// status of each simulation's last call
const int* mjb_status(const mjBatch* b) {
  return b->status.data();
}


// error message of simulation sim's last call
const char* mjb_error(const mjBatch* b, int sim) {
  if (sim < 0 || sim >= b->nsim) return "";
  return b->errors[sim].c_str();
}
