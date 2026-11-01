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

import 'jasmine';

import { MainModule, DoubleBuffer } from "../dist/mujoco_wasm_benchmark"
import loadMujoco from "../dist/mujoco_wasm_benchmark.js"

describe('MuJoCo WASM Benchmark Tests', () => {
  let mujoco: MainModule;

  function isNumberArraySorted(arr: number[]): boolean {
    for (let i = 0; i < arr.length - 1; i++) {
      if (arr[i] > arr[i + 1]) {
        return false;
      }
    }
    return true;
  }

  function verifySorted(state: any, kind: string): void {
    let array: number[] = [];
    if (kind === 'NumberArray') {
      array = state as number[];
    } else {
      const bufferView = (state as DoubleBuffer).GetView();
      array = Array.from(bufferView) as number[];
    }

    const sorted = isNumberArraySorted(array);
    if (!sorted) {
      console.error(`  Verification failed for ${kind}. Array was ${array}`);
    }
  }

  function runBenchmark(iterations: number, items: number, kind: string) {
    let state: any;
    let func: (state: any) => any;
    if (kind === 'NumberArray') {
      state = new Array(items).map(() => Math.random())
      func = (state) => mujoco.SortNumberArray(state);
    } else if (kind === 'DoubleBuffer') {
      state = mujoco.DoubleBuffer.FromArray(
          new Array(items).map(() => Math.random()));
      func = (state) => mujoco.SortDoubleBuffer(state);
    } else {
      throw new Error(`Unsupported benchmark type: ${kind}`);
    }

    // Warmup JIT compiler to get stable results
    for (let i = 0; i < 100; i++) {
      const sortedState = func(state);
      verifySorted(sortedState, kind);
    }

    // Measurement
    const totalStartTime = performance.now();
    for (let i = 0; i < iterations; i++) {
      func(state);
    }
    const totalEndTime = performance.now();

    if (kind === 'DoubleBuffer') {
      state.delete();
    }

    // Report results
    const totalTime = totalEndTime - totalStartTime;
    const avgTimeMilliseconds = totalTime / iterations;
    console.log(`Benchmark: "Sort ${kind} ${iterations} iterations with ${
        items} items" - AVG time per call: ${
        avgTimeMilliseconds.toFixed(2)} ms`);
    return {
      totalTime, avgTimeMilliseconds,
    }
  }

  beforeAll(async () => {
    mujoco = await loadMujoco();
  }, 60_000);

  it('should benchmark NumberArray and DoubleBuffer and compare results',
     () => {
       const na1 = runBenchmark(100, 1_000, 'NumberArray');
       const na2 = runBenchmark(100, 400_000, 'NumberArray');

       const db1 = runBenchmark(100, 1_000, 'DoubleBuffer');
       const db2 = runBenchmark(100, 400_000, 'DoubleBuffer');

       // The actual time should be much faster than that but the intention of
       // the check is to catch huge regressions without a flakey test
       const _100ms = 100;

       expect(db2.avgTimeMilliseconds).toBeLessThan(_100ms);

       expect(db1.totalTime).toBeLessThan(na1.totalTime);
       expect(db2.totalTime).toBeLessThan(na2.totalTime);

       expect(db1.avgTimeMilliseconds).toBeLessThan(na1.avgTimeMilliseconds);
       expect(db2.avgTimeMilliseconds).toBeLessThan(na2.avgTimeMilliseconds);
     });
});
