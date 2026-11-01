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

import {MainModule, MjContact, MjContactVec, MjData, MjLROpt, MjModel,
MjOption, MjsGeom, MjSolverStat, MjSpec, MjStatistic, MjTimerStat, MjvCamera,
MjvFigure, MjvGeom, MjvGLCamera, MjvLight, MjvOption, MjvPerturb, MjvScene,
MjWarningStat, MjVFS, Uint8Buffer} from '../dist/mujoco.js';

import loadMujoco from '../dist/mujoco.js'

function assertExists<T>(value: T | null | undefined, message?: string):
asserts value is T {
  if (value === null || value === undefined) {
    throw new Error(message ?? 'Expected value to be defined.');
  }
}


// Corresponds to bindings_test.py:TEST_XML
const TEST_XML = `
<mujoco model="test">
  <compiler coordinate="local" angle="radian" eulerseq="xyz"/>
  <size nkey="2"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global fovy="50" />
    <quality shadowsize="51" />
  </visual>
  <worldbody>
    <geom name="myplane" type="plane" size="10 10 1" user="1 2 3"/>
    <body name="mybox" pos="0 0 0.1">
      <geom name="mybox" type="box" size="0.1 0.1 0.1" mass="0.25"/>
      <freejoint name="myfree"/>
    </body>
    <body name="myhinge-body" pos="0 0 1">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <site pos="0 0 -1" name="mysite" type="sphere"/>
      <joint name="myhinge" type="hinge" axis="0 1 0" damping="1"/>
    </body>
    <body name="myball-body" pos="2 0 1">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="myball" type="ball"/>
    </body>
    <body name="mocap-body" mocap="true" pos="42 0 42">
      <geom name="mocap-sphere" type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="myactuator" joint="myhinge"/>
  </actuator>
  <sensor>
    <jointvel name="myjointvel" joint="myhinge"/>
    <accelerometer name="myaccelerometer" site="mysite"/>
  </sensor>
</mujoco>
`;

type TypedArray =|Int8Array|Uint8Array|Uint8ClampedArray|Int16Array|Uint16Array|
    Int32Array|Uint32Array|Float32Array|Float64Array;

function norm(arr: number[]): number {
  return Math.sqrt(arr.reduce((acc, val) => acc + val * val, 0));
}

function expectArraysClose(arr1: any, arr2: TypedArray, precision = 1) {
  expect(arr1.length).toEqual(arr2.length);
  for (let i = 0; i < arr1.length; i++) {
    expect(arr1[i]).toBeCloseTo(arr2[i], precision);
  }
}

function expectArraysEqual(arr1: any, arr2: TypedArray) {
  expect(arr1.length).toEqual(arr2.length);
  for (let i = 0; i < arr1.length; i++) {
    expect(arr1[i]).toEqual(arr2[i]);
  }
}

describe('MuJoCo WASM Bindings', () => {
  let mujoco: MainModule;
  let model: MjModel|null = null;
  let data: MjData|null = null;

  beforeAll(async () => {
    mujoco = await loadMujoco();
  }, 60_000);

  function unlinkXMLFile(filename: string) {
    try {
      (mujoco as any).FS.unlink(filename);
    } catch (e) {
      console.warn(`Failed to unlink temporary XML file: ${e}`);
    }
  }

  function writeXMLFile(filename: string, xmlContent: string) {
    try {
      (mujoco as any).FS.writeFile(filename, xmlContent);
    } catch (e) {
      throw new Error(`Failed to write temporary XML file: ${e}`);
    }
  }

  beforeEach(() => {
    const tempXmlFilename = '/tmp/model.xml';

    writeXMLFile(tempXmlFilename, TEST_XML);

    model = mujoco.MjModel!.mj_loadXML(tempXmlFilename);
    if (!model) {
      unlinkXMLFile(tempXmlFilename);
      throw new Error('Failed to load model from XML');
    }

    unlinkXMLFile(tempXmlFilename);

    data = new mujoco.MjData(model);
    if (!data) {
      throw new Error('Failed to create data from model');
    }
  });

  afterEach(() => {
    model?.delete();
    data?.delete();
  });

  describe('Buffer API', () => {
    it('should construct from an element count', () => {
      const buf = new mujoco.DoubleBuffer(5);
      expect(buf.GetElementCount()).toBe(5);
      expect(buf.GetPointer()).toBeDefined();
      expect(buf.GetView()).toBeDefined();
    });

    it('should construct from a Javascript array', () => {
      const array: number[] = [0, 1, 4, 9, 16];
      const buf = mujoco.DoubleBuffer.FromArray(array);
      expect(buf.GetElementCount()).toBe(5);
      expect(buf.GetPointer()).toBeDefined();
      expect(buf.GetView()).toBeDefined();
      expectArraysClose(buf.GetView(), new Float64Array(array));
    });

    it('should construct from Float64Array', () => {
      const array = new Float64Array([1 / 11, 7 / 11]);
      const buf = mujoco.DoubleBuffer.FromArray(array);
      expect(buf.GetElementCount()).toBe(2);
      expect(buf.GetPointer()).toBeDefined();
      expect(buf.GetView()).toBeDefined();
      expectArraysClose(buf.GetView(), array);
    });
  });

  describe('mj_addM', () => {
    let simpleModel: MjModel|null = null;
    let simpleData: MjData|null = null;
    const tempXmlFilename = '/tmp/simple_model.xml';
    // A simpler model with nv=1 and nM=1 to avoid WASM binding bugs.
    const simpleXmlContent = `
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>`;

    beforeEach(() => {
      writeXMLFile(tempXmlFilename, simpleXmlContent);
      simpleModel = mujoco.MjModel!.mj_loadXML(tempXmlFilename);
      assertExists(simpleModel);
      simpleData = new mujoco.MjData(simpleModel);
      assertExists(simpleData);
    });

    afterEach(() => {
      simpleModel?.delete();
      simpleData?.delete();
      unlinkXMLFile(tempXmlFilename);
    });

    it('should compute the sparse inertia matrix', () => {
      const nM = simpleModel!.nM;
      const dstSparse = new mujoco.DoubleBuffer(nM);
      try {
        mujoco.mj_forward(simpleModel!, simpleData!);
        mujoco.mj_addM(
            simpleModel!, simpleData!, dstSparse, simpleModel!.M_rownnz,
            simpleModel!.M_rowadr, simpleModel!.M_colind);

        expect(dstSparse.GetView().length).toBe(1);
        expect(dstSparse.GetView()[0]).toBeCloseTo(1.0);
      } finally {
        dstSparse.delete();
      }
    });

    it('should throw an error for incorrect sparse matrix dimensions', () => {
      const nM = simpleModel!.nM;
      const dstSparse = new mujoco.DoubleBuffer(nM + 1);
      try {
        expect(
            () => mujoco.mj_addM(
                simpleModel!, simpleData!, dstSparse, simpleModel!.M_rownnz,
                simpleModel!.M_rowadr, simpleModel!.M_colind))
            .toThrowError(
                'MuJoCo Error: [mj_addM] dst must have size 1, got 2');
      } finally {
        dstSparse.delete();
      }
    });

    it('should compute the sparse inertia matrix with null pointers', () => {
      const nM = simpleModel!.nM;
      const dstSparse = new mujoco.DoubleBuffer(nM);
      try {
        mujoco.mj_forward(simpleModel!, simpleData!);
        mujoco.mj_addM(simpleModel!, simpleData!, dstSparse, null, null, null);

        expect(dstSparse.GetView().length).toBe(1);
      } finally {
        dstSparse.delete();
      }
    });
  });

  it('should compute geom distance without returning fromto', () => {
    mujoco.mj_forward(model!, data!);
    const dist = mujoco.mj_geomDistance(model!, data!, 0, 2, 200, null);
    expect(dist).toEqual(41.9);
  });

  it('should handle box QP solver with null optionals', () => {
    const n = 5;
    const res = new mujoco.DoubleBuffer(n);
    const r = new mujoco.DoubleBuffer(n * (n + 7));
    const h = new mujoco.DoubleBuffer(n * n);
    const g = mujoco.DoubleBuffer.FromArray(new Array(n).fill(1));
    try {
      for (let i = 0; i < n; i++) {
        h.GetView()[i * (n + 1)] = 1;
      }
      const rank = mujoco.mju_boxQP(
          res, r, null, h.GetView(), g.GetView(), null as any, null as any);
      expect(rank).toBeGreaterThan(-1);
    } finally {
      res.delete();
      r.delete();
      h.delete();
      g.delete();
    }
  });

  // Corresponds to engine_core_constraint_test.cc:TEST_F(CoreConstraintTest,
  // ConstraintUpdateImpl)
  it('should correctly compute constraint cost', () => {
    const xmlString = `
<mujoco>
  <option>
    <flag island="enable"/>
  </option>

  <default>
    <geom size=".1"/>
  </default>

  <visual>
    <headlight diffuse=".9 .9 .9"/>
  </visual>

  <worldbody>

    <body>
      <joint type="slide" axis="0 0 1" range="0 1" limited="true"/>
      <geom/>
    </body>

    <body pos=".25 0 0">
      <joint type="slide" axis="1 0 0"/>
      <geom/>
    </body>

    <body pos="0 0 0.25">
      <joint type="slide" axis="0 0 1"/>
      <geom/>
      <body pos="0 -.15 0">
        <joint name="hinge1" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
        <body pos="-.2 0 0">
          <joint axis="0 1 0"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
        </body>
      </body>
    </body>

    <body pos=".5 0 0">
      <joint type="slide" axis="0 0 1" frictionloss="15"/>
      <geom type="box" size=".08 .08 .02" euler="0 10 0"/>
    </body>

    <body pos="-.5 0 0">
      <joint axis="0 1 0" frictionloss=".01"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
    </body>

    <body pos="0 0 .5">
      <joint name="hinge2" axis="0 1 0"/>
      <geom type="box" size=".08 .02 .08"/>
    </body>

    <body pos=".5 0 .1">
      <freejoint/>
      <geom type="box" size=".03 .03 .03" pos="0.01 0.01 0.01"/>
    </body>

    <site name="0" pos="-.45 -.05 .35"/>
    <body pos="-.5 0 .3" name="connect">
      <freejoint/>
      <geom type="box" size=".05 .05 .05"/>
      <site name="1" pos=".05 -.05 .05"/>
    </body>
  </worldbody>

  <equality>
    <joint joint1="hinge1" joint2="hinge2"/>
    <connect body1="connect" body2="world" anchor="-.05 -.05 .05"/>
    <connect site1="0" site2="1"/>
  </equality>
</mujoco>
    `;
    const tempXmlFilename = '/tmp/model_c.xml';
    writeXMLFile(tempXmlFilename, xmlString);
    const model = mujoco.MjModel!.mj_loadXML(tempXmlFilename);
    expect(model).not.toBeNull();
    const data = new mujoco.MjData(model!);
    expect(data).not.toBeNull();
    unlinkXMLFile(tempXmlFilename);
    try {
      mujoco.mj_resetData(model, data!);
      let steps = 0;
      while (data!.ncon === 0 && steps < 100) {
        mujoco.mj_step(model!, data!);
        steps++;
      }
      mujoco.mj_forward(model!, data!);
      const res = new mujoco.DoubleBuffer(data!.nefc);
      mujoco.mj_mulJacVec(model!, data!, res, data!.qacc);
      mujoco.mju_subFrom(res, data!.efc_aref);
      const cost = mujoco.DoubleBuffer.FromArray([0]);
      mujoco.mj_constraintUpdate(
          model!, data!, res.GetView(), cost, /*flg_coneHessian=*/ 1);

      expect(cost.GetView()[0]).toBeCloseTo(3357.584);

      res.delete();
      cost.delete();
    } finally {
      data!.delete();
      model!.delete();
    }
  });

  it('should compute body jacobian', () => {
    mujoco.mj_forward(model!, data!);
    const bodyId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_BODY.value, 'mybox');
    const point = [0.1, 0.2, 0.3];
    const jacp = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacr = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_jac(model!, data!, jacp, jacr, point, bodyId);
      expect(norm(jacp.GetView())).toBeGreaterThan(0);
      expect(norm(jacr.GetView())).toBeGreaterThan(0);
      expect(() => mujoco.mj_jac(model!, data!, null, null, point, bodyId))
          .not.toThrow();
    } finally {
      jacp.delete();
      jacr.delete();
    }
  });

  it('should compute body frame jacobian', () => {
    mujoco.mj_forward(model!, data!);
    const bodyId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_BODY.value, 'mybox');
    const jacp = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacr = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_jacBody(model!, data!, jacp, jacr, bodyId);
      expect(norm(jacp.GetView())).toBeGreaterThan(0);
      expect(norm(jacr.GetView())).toBeGreaterThan(0);
      expect(() => mujoco.mj_jacBody(model!, data!, null, null, bodyId))
          .not.toThrow();
    } finally {
      jacp.delete();
      jacr.delete();
    }
  });

  it('should compute body CoM jacobian', () => {
    mujoco.mj_forward(model!, data!);
    const bodyId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_BODY.value, 'mybox');
    const jacp = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacr = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_jacBodyCom(model!, data!, jacp, jacr, bodyId);
      expect(norm(jacp.GetView())).toBeGreaterThan(0);
      expect(norm(jacr.GetView())).toBeGreaterThan(0);
      expect(() => mujoco.mj_jacBodyCom(model!, data!, null, null, bodyId))
          .not.toThrow();
    } finally {
      jacp.delete();
      jacr.delete();
    }
  });

  it('should compute geom jacobian', () => {
    mujoco.mj_forward(model!, data!);
    const geomId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_GEOM.value, 'mybox');
    const jacp = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacr = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_jacGeom(model!, data!, jacp, jacr, geomId);
      expect(norm(jacp.GetView())).toBeGreaterThan(0);
      expect(norm(jacr.GetView())).toBeGreaterThan(0);
      expect(() => mujoco.mj_jacGeom(model!, data!, null, null, geomId))
          .not.toThrow();
    } finally {
      jacp.delete();
      jacr.delete();
    }
  });

  it('should compute point-axis jacobian', () => {
    mujoco.mj_forward(model!, data!);
    const bodyId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_BODY.value, 'mybox');
    const point = [0.1, 0.2, 0.3];
    const axis = [0, 0, 1];
    const jacPoint = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacAxis = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_jacPointAxis(
          model!, data!, jacPoint, jacAxis, point, axis, bodyId);
      expect(norm(jacPoint.GetView())).toBeGreaterThan(0);
      expect(norm(jacAxis.GetView())).toBeGreaterThan(0);
      expect(
          () => mujoco.mj_jacPointAxis(
              model!, data!, null, null, point, axis, bodyId))
          .not.toThrow();
    } finally {
      jacPoint.delete();
      jacAxis.delete();
    }
  });

  it('should compute jacobian time derivative', () => {
    mujoco.mj_forward(model!, data!);
    const bodyId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_BODY.value, 'mybox');
    const point = [0.1, 0.2, 0.3];
    const jacp = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacr = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_jacDot(model!, data!, jacp, jacr, point, bodyId);
      expect(jacp.GetView().length).toBe(3 * model!.nv);
      expect(jacr.GetView().length).toBe(3 * model!.nv);
      expect(() => mujoco.mj_jacDot(model!, data!, null, null, point, bodyId))
          .not.toThrow();
    } finally {
      jacp.delete();
      jacr.delete();
    }
  });

  it('should apply external force and torque', () => {
    const force = [1, .5, 1];
    const torque = [.3, 1, .22];
    const point = [0, 0, 0];
    const bodyId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_BODY.value, 'mybox');
    const qfrcTarget = new mujoco.DoubleBuffer(model!.nv);
    try {
      mujoco.mj_forward(model!, data!);
      mujoco.mj_applyFT(
          model!, data!, force, torque, point, bodyId, qfrcTarget);
      expect(norm(qfrcTarget.GetView())).toBeGreaterThan(0);

      qfrcTarget.GetView().fill(0);
      mujoco.mj_applyFT(
          model!, data!, null as any, null as any, point, bodyId, qfrcTarget);
      expect(norm(qfrcTarget.GetView())).toEqual(0);
    } finally {
      qfrcTarget.delete();
    }
  });

  it('should compute finite-differenced transition matrices', () => {
    const eps = 1e-6;
    const flg_centered = false;
    const dim = 2 * model!.nv + model!.na;
    const A = new mujoco.DoubleBuffer(dim * dim);
    const B = new mujoco.DoubleBuffer(dim * model!.nu);
    const C = new mujoco.DoubleBuffer(model!.nsensordata * dim);
    const D = new mujoco.DoubleBuffer(model!.nsensordata * model!.nu);
    try {
      mujoco.mjd_transitionFD(model!, data!, eps, flg_centered, A, B, C, D);
      expect(norm(A.GetView())).toBeGreaterThan(0);
      expect(norm(B.GetView())).toBeGreaterThan(0);
      expect(norm(C.GetView())).toBeGreaterThan(0);
      expect(
          () => mujoco.mjd_transitionFD(
              model!, data!, eps, flg_centered, null, null, null, null))
          .not.toThrow();
    } finally {
      A.delete();
      B.delete();
      C.delete();
      D.delete();
    }
  });

  it('should solve a system of linear equations', () => {
    const n = 2;
    const mat = [4, 1, 1, 3];
    const vec = [1, 2];
    const res = new mujoco.DoubleBuffer(n);
    const expected = new Float64Array([1 / 11, 7 / 11]);
    const matFactor = mujoco.DoubleBuffer.FromArray(mat);
    try {
      const rank = mujoco.mju_cholFactor(matFactor, 1e-9);
      expect(rank).toBe(n);
      mujoco.mju_cholSolve(res, matFactor.GetView(), vec);
      const result = res.GetView();
      expect(result[0]).toBeCloseTo(expected[0]);
      expect(result[1]).toBeCloseTo(expected[1]);
    } finally {
      res.delete();
      matFactor.delete();
    }
  });

  it('should update a Cholesky factorization', () => {
    const n = 2;
    const mat = [4, 1, 1, 3];
    const x = [1, 1];
    const matFactor = mujoco.DoubleBuffer.FromArray(mat);
    const xBuf = mujoco.DoubleBuffer.FromArray(x);
    const res = new mujoco.DoubleBuffer(n);
    try {
      // Factorize original matrix.
      mujoco.mju_cholFactor(matFactor, 1e-9);

      // Update factorization with x.
      const flg_plus = 1;
      mujoco.mju_cholUpdate(matFactor, xBuf.GetView(), flg_plus);

      // Solve a system with the updated factor to verify.
      // A' = A + x*x' = [[5, 2], [2, 4]]
      // A' * y = vec => [[5, 2], [2, 4]] * y = [7, 6]
      // Solution is y = [1, 1].
      const vec = [7, 6];
      mujoco.mju_cholSolve(res, matFactor.GetView(), vec);
      const result = res.GetView();
      const expectedSolution = new Float64Array([1, 1]);
      expect(result[0]).toBeCloseTo(expectedSolution[0]);
      expect(result[1]).toBeCloseTo(expectedSolution[1]);
    } finally {
      matFactor.delete();
      xBuf.delete();
      res.delete();
    }
  });

  it('should multiply a transposed matrix by another matrix', () => {
    const r1 = 3, c1 = 2, c2 = 2;
    const mat1 = mujoco.DoubleBuffer.FromArray([1, 4, 2, 5, 3, 6]);
    const mat2 = mujoco.DoubleBuffer.FromArray([7, 8, 9, 10, 11, 12]);
    const res = new mujoco.DoubleBuffer(c1 * c2);
    const expected = new Float64Array([58, 64, 139, 154]);

    try {
      mujoco.mju_mulMatTMat(res, mat1.GetView(), mat2.GetView(), r1, c1, c2);
      expectArraysClose(res.GetView(), expected);
    } finally {
      mat1.delete();
      mat2.delete();
      res.delete();
    }
  });

  it('should throw an error because of incompatible matrix sizes', () => {
    const r1 = 3, c1 = 2, c2 = 2;
    const mat1 = mujoco.DoubleBuffer.FromArray([1, 4, 2, 5, 3, 6]);
    const mat2 = mujoco.DoubleBuffer.FromArray([7, 8, 9, 10, 11]);
    const res = new mujoco.DoubleBuffer(c1 * c2);
    try {
      expect(
          () => mujoco.mju_mulMatTMat(
              res, mat1.GetView(), mat2.GetView(), r1, c1, c2))
          .toThrowError(
              'MuJoCo Error: [mju_mulMatTMat] mat2 must have size 6, got 5');
    } finally {
      mat1.delete();
      mat2.delete();
      res.delete();
    }
  });

  it('should convert a dense matrix to sparse and return non-zero count',
     () => {
       const nr = 2;
       const nc = 3;
       const mat = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
       const rownnz = new mujoco.IntBuffer(nr);
       const rowadr = new mujoco.IntBuffer(nr);
       const colind = new mujoco.IntBuffer(nc);
       const res = new mujoco.DoubleBuffer(nc);
       try {
         const nnz =
             mujoco.mju_dense2sparse(res, mat, nr, nc, rownnz, rowadr, colind);
         expectArraysEqual(res.GetView(), new Float64Array([1.0, 2.0, 3.0]));
         expectArraysEqual(rownnz.GetView(), new Int32Array([1, 2]));
         expectArraysEqual(rowadr.GetView(), new Int32Array([0, 1]));
         expectArraysEqual(colind.GetView(), new Int32Array([1, 0, 2]));
       } finally {
         res.delete();
         rownnz.delete();
         rowadr.delete();
         colind.delete();
       }
     });

  it('should convert a sparse matrix to a dense matrix', () => {
    const nr = 2;
    const nc = 3;
    const mat = [1.0, 2.0, 3.0];
    const rownnz = [1, 2];
    const rowadr = [0, 1];
    const colind = [1, 0, 2];
    const res = new mujoco.DoubleBuffer(nr * nc);
    const expected = new Float64Array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    try {
      mujoco.mju_sparse2dense(res, mat, nr, nc, rownnz, rowadr, colind);
      expectArraysEqual(res.GetView(), expected);
    } finally {
      res.delete();
    }
  });

  it('should throw an error when mju_eye is called with a null argument', () => {
    expect(() => {
      mujoco.mju_eye(null as any);
    })
        .toThrowError(
            'MuJoCo Error: [mju_eye] Invalid argument. Expected a TypedArray or WasmBuffer, got null.');
  });

  it('should return undefined', () => {
    const spec = mujoco.parseXMLString(TEST_XML);
    const body = mujoco.mjs_findBody(spec, 'some_name_that_doesnt_exist');
    expect(body).toBeUndefined();
    body?.delete();
    spec?.delete();
  });

  it('should check constants values', () => {
    expect(mujoco.mjNEQDATA).toBe(12);
    expect(mujoco.mjDISABLESTRING).toEqual([
      'Constraint',   'Equality',  'Frictionloss', 'Limit',     'Contact',
      'Spring',       'Damper',    'Gravity',      'Clampctrl', 'Warmstart',
      'Filterparent', 'Actuation', 'Refsafe',      'Sensor',    'Midphase',
      'Eulerdamp',    'AutoReset', 'NativeCCD',    'Island',    'MultiCCD',
    ]);
    expect(mujoco.mjRNDSTRING).toEqual([
      ['Shadow', '1', 'S'], ['Wireframe', '0', 'W'], ['Reflection', '1', 'R'],
      ['Additive', '0', 'L'], ['Skybox', '1', 'K'], ['Fog', '0', 'G'],
      ['Haze', '1', '/'], ['Depth', '0', ''], ['Segment', '0', ','],
      ['Id Color', '0', ''], ['Cull Face', '1', ''],
      ['Post Processing', '1', ''],
    ]);
    expect(mujoco.mjFRAMESTRING.length).toEqual(mujoco.mjtFrame.mjNFRAME.value);
    expect(mujoco.mjVISSTRING.length)
        .toEqual(mujoco.mjtVisFlag.mjNVISFLAG.value);
    expect(mujoco.mjVISSTRING[mujoco.mjtVisFlag.mjVIS_INERTIA.value]).toEqual([
      'Inertia', '0', 'I'
    ]);
  });

  it('should create a spec from XML', () => {
    const spec = mujoco.parseXMLString(TEST_XML);
    try {
      expect(spec).toBeDefined();
      expect(spec!.modelname).toEqual('test');
    } finally {
      spec?.delete();
    }
  });

  it('should return the max value', () => {
    expect(mujoco.mju_max(10, 2)).toEqual(10);
  });

  // Corresponds to bindings_test.py:test_mju_rotVecQuat
  it('should rotate a vector by a quaternion', () => {
    const res = mujoco.DoubleBuffer.FromArray([0, 0, 0]);
    const vec = [1, 0, 0];
    const angle = 7 * Math.PI / 12;
    const quat = [Math.cos(angle), 0, 0, Math.sin(angle)];
    const expected = new Float64Array([-0.8660377, -0.5, 0]);
    try {
      mujoco.mju_rotVecQuat(res, vec, quat);
      expectArraysClose(res.GetView(), expected, 4);
    } finally {
      res.delete();
    }
  });

  it('should get correct geom name', () => {
    const spec = mujoco.parseXMLString(TEST_XML);
    const geomEl =
        mujoco.mjs_findElement(spec, mujoco.mjtObj.mjOBJ_GEOM, 'myplane');
    try {
      assertExists(geomEl);
      const geom = mujoco.mjs_asGeom(geomEl);
      assertExists(geom);
      const geomName = mujoco.mjs_getName(geom.element);
      expect(geomName).toEqual('myplane');
      mujoco.mjs_setName(geom.element, 'myplane2');
      expect(mujoco.mjs_getName(geom.element)).toEqual('myplane2');
    } finally {
      spec?.delete();
    }
  });

  it('should override geom userdata', () => {
    const spec = mujoco.parseXMLString(TEST_XML);
    const geomEl =
        mujoco.mjs_findElement(spec, mujoco.mjtObj.mjOBJ_GEOM, 'myplane');
    try {
      assertExists(geomEl);
      const geom = mujoco.mjs_asGeom(geomEl);
      assertExists(geom);
      geom.userdata.set(0, 10);
      expect(geom.userdata.get(0)).toEqual(10);
    } finally {
      spec.delete();
    }
  })

  it('should override model geom_rgba', () => {
    const newValues = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    model!.geom_rgba.set(newValues);
    const expected = new Float32Array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1]);
    expectArraysClose(model!.geom_rgba, expected);
  })

  it('should add a scaled vector to another vector', () => {
    const res = mujoco.DoubleBuffer.FromArray([1, 2, 3]);
    const vec = [4, 5, 6];
    const scale = 2;
    const expected = new Float64Array([9, 12, 15]);
    try {
      mujoco.mju_addToScl(res, vec, scale);
      expectArraysClose(res.GetView(), expected, 4);
    } finally {
      res.delete();
    }
  });

  it('should throw an error when mju_addToScl is called with incompatible sizes',
     () => {
       const res = mujoco.DoubleBuffer.FromArray([1, 2, 3]);
       const vec = [4, 5];
       const scale = 2;
       try {
         expect(() => {
           mujoco.mju_addToScl(res, vec, scale);
         })
             .toThrowError(
                 'MuJoCo Error: [mju_addToScl] res and vec must have equal size, got 3 and 2');
       } finally {
         res.delete();
       }
     });

  it('should sort an array with insertion sort', () => {
    const arr = mujoco.DoubleBuffer.FromArray([5, 2, 8, 1, 9]);
    const expected = new Float64Array([1, 2, 5, 8, 9]);
    try {
      mujoco.mju_insertionSort(arr);
      expectArraysEqual(arr.GetView(), expected);
    } finally {
      arr.delete();
    }
  });

  it('should find the attached spec', () => {
    const bXml = `
    <mujoco>
      <worldbody>
        <body name="b" pos="0 0 0.1">
          <geom rgba="0 .9 0 1" name="attached_geom_b" type="box" size="0.1 0.1 0.1" mass="0.25"/>
          <freejoint name="bfree"/>
        </body>
      </worldbody>
    </mujoco>
    `;
    const xmlWithAttachedSpec = `
    <mujoco>
      <asset>
        <model name="b" file="b.xml" />
      </asset>

      <worldbody>
        <attach model="b" body="b" prefix="b" />
      </worldbody>
    </mujoco>
    `;
    const mainXmlFilename = 'main.xml';
    const bXmlFilename = 'b.xml';
    writeXMLFile(mainXmlFilename, xmlWithAttachedSpec);
    writeXMLFile(bXmlFilename, bXml);
    const spec = mujoco.parseXMLString(xmlWithAttachedSpec);
    const attachedSpec = mujoco.mjs_findSpec(spec, 'b');

    try {
      assertExists(attachedSpec);
      const geomEl = mujoco.mjs_findElement(
          attachedSpec, mujoco.mjtObj.mjOBJ_GEOM, 'attached_geom_b');
      assertExists(geomEl);
      const geom = mujoco.mjs_asGeom(geomEl);
      expect(geom).toBeDefined();
      expect(mujoco.mjs_getName(geom!.element)).toEqual('attached_geom_b');
    } finally {
      attachedSpec?.delete();
      spec?.delete();
      unlinkXMLFile(mainXmlFilename);
      unlinkXMLFile(bXmlFilename);
    }
  });

  // Corresponds to bindings_test.py:test_load_xml_can_handle_name_clash
  it('should handle name clashes when loading XML with includes', () => {
    const xml1 = `
      <mujoco>
        <worldbody>
          <geom name="plane" type="plane" size="1 1 1"/>
          <include file="model_.xml"/>
          <include file="model__.xml"/>
        </worldbody>
      </mujoco>
    `;
    const xml2 = `<mujoco><geom name="box" type="box" size="1 1 1"/></mujoco>`;
    const xml3 = `<mujoco><geom name="ball" type="sphere" size="1"/></mujoco>`;

    const modelXmlFilename = 'model.xml';
    const model1XmlFilename = 'model_.xml';
    const model2XmlFilename = 'model__.xml';

    writeXMLFile(modelXmlFilename, xml1);
    writeXMLFile(model1XmlFilename, xml2);
    writeXMLFile(model2XmlFilename, xml3);

    const model = mujoco.MjModel!.mj_loadXML(modelXmlFilename);

    try {
      expect(model).toBeDefined();
      expect(mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_GEOM.value, 'plane'))
          .toBe(0);
      expect(mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_GEOM.value, 'box'))
          .toBe(1);
      expect(mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_GEOM.value, 'ball'))
          .toBe(2);
    } finally {
      model?.delete();
      unlinkXMLFile(modelXmlFilename);
      unlinkXMLFile(model1XmlFilename);
      unlinkXMLFile(model2XmlFilename);
    }
  });

  // Corresponds to bindings_test.py:test_can_read_array
  it('should read an array from the model', () => {
    const expected =
        new Float64Array([0, 0, 0, 0, 0, 0.1, 0, 0, 1, 2, 0, 1, 42, 0, 42]);
    const bodyPos = new Float64Array(model!.body_pos);
    expectArraysEqual(bodyPos, expected);
  });

  // Corresponds to bindings_test.py:test_can_set_array
  it('should set an array from the data', () => {
    const value = 0.12345;
    data!.qpos.fill(value);
    const expected = new Float64Array(data!.qpos.length).fill(value);
    expectArraysEqual(data!.qpos, expected);
  });

  // Corresponds to bindings_test.py:test_array_is_a_view
  it('should check that array is a view', () => {
    const qposRef = data!.qpos;
    const value = 0.789;
    data!.qpos.fill(value);
    const expected = new Float64Array(data!.qpos.length).fill(value);
    expectArraysEqual(qposRef, expected);
  });

  it('should read and write mjtBool arrays as Uint8Array', () => {
    expect(data!.eq_active).toBeInstanceOf(Uint8Array);
    const jntLimited = model!.jnt_limited;
    expect(jntLimited).toBeInstanceOf(Uint8Array);
    expectArraysEqual(jntLimited, new Uint8Array([0, 0, 0]));
    jntLimited[1] = 1;
    expectArraysEqual(model!.jnt_limited, new Uint8Array([0, 1, 0]));
  });

  // Corresponds to bindings_test.py:test_mjmodel_can_read_and_write_opt
  it('should read and write MjOption', () => {
    expect(model!.opt.timestep).toEqual(0.002);
    expectArraysEqual(model!.opt.gravity, new Float64Array([0, 0, -9.81]));

    const optRef = model!.opt;
    model!.opt.timestep = 0.001;
    expect(optRef.timestep).toEqual(0.001);

    const gravityRef = optRef.gravity;
    model!.opt.gravity[1] = 0.1;
    expectArraysEqual(gravityRef, new Float64Array([0, 0.1, -9.81]));

    model!.opt.gravity.fill(0.2);
    expectArraysEqual(gravityRef, new Float64Array([0.2, 0.2, 0.2]));
  });

  // Corresponds to bindings_test.py:test_mjmodel_can_read_and_write_stat
  it('should read and write MjStat', () => {
    expect(model!.stat.meanmass).not.toEqual(0);

    const statRef = model!.stat;
    model!.stat.meanmass = 1.2;
    expect(statRef.meanmass).toEqual(1.2);
  });

  // Corresponds to bindings_test.py:test_mjmodel_can_read_and_write_vis
  it('should read and write MjVis', () => {
    expect(model!.vis.quality.shadowsize).toEqual(51);

    const visRef = model!.vis;
    model!.vis.quality.shadowsize = 100;
    expect(visRef.quality.shadowsize).toEqual(100);
  });

  // Corresponds to bindings_test.py:test_mjmodel_can_access_names_directly
  it('should access names directly from the model', () => {
    const modelName = new TextDecoder().decode(
        model!.names.slice(0, model!.names.indexOf(0)));
    expect(modelName).toEqual('test');

    const startGeomNameIndex = model!.name_geomadr[0];
    const endGeomNameIndex = model!.names.indexOf(0, startGeomNameIndex);
    const geomName = new TextDecoder().decode(
        model!.names.slice(startGeomNameIndex, endGeomNameIndex));
    expect(geomName).toEqual('myplane');
  });

  // Corresponds to bindings_test.py:test_mjmodel_names_doesnt_copy
  it('should not copy names when accessing them multiple times', () => {
    const names1 = model!.names;
    const names2 = model!.names;
    expect(names1).toEqual(names2);
  });

  // Corresponds to bindings_test.py:test_mjoption_can_make_default
  it('should create a default MjOption', () => {
    const opt = new mujoco.MjOption();
    expect(opt.timestep).toEqual(0.002);
    expectArraysEqual(opt.gravity, new Float64Array([0, 0, -9.81]));
  });

  // Corresponds to bindings_test.py:test_mjoption_can_copy
  it('should copy MjOption', () => {
    const opt1 = new mujoco.MjOption();
    opt1.timestep = 0.001;
    opt1.gravity.set([2, 2, 2]);

    const opt2 = opt1.copy();
    expect(opt2.timestep).toEqual(0.001);
    expectArraysEqual(opt2.gravity, new Float64Array([2, 2, 2]));

    opt1.timestep = 0.005;
    opt1.gravity.set([5, 5, 5]);
    expect(opt2.timestep).toEqual(0.001);
    expectArraysEqual(opt2.gravity, new Float64Array([2, 2, 2]));
  });

  // Corresponds to bindings_test.py:test_mjdata_can_read_warning_array
  it('should read warning array from MjData', () => {
    expect(data!.warning.size()).toEqual(mujoco.mjtWarning.mjNWARNING.value);
    data!.qpos[0] = NaN;
    mujoco.mj_checkPos(model!, data!);
    expect(data!.warning.get(mujoco.mjtWarning.mjWARN_BADQPOS.value)!.number)
        .toEqual(1);
  });

  // Corresponds to bindings_test.py:test_mjcontact_can_copy
  it('should copy MjContact', () => {
    mujoco.mj_forward(model!, data!!);
    const contacts: MjContactVec = data!.contact;
    const originalContact = contacts.get(0)!;
    const originalPos = new Float64Array(originalContact.pos);
    const copiedContact = originalContact.copy();
    copiedContact.delete();

    expectArraysClose(originalContact.pos, originalPos);
    originalContact.delete();
  });

  // Corresponds to bindings_test.py:test_mj_step
  it('should step the simulation forward', () => {
    const displacement = 0.55;
    data!.qpos[2] += displacement;
    mujoco.mj_forward(model!, data!);

    const gravity = -model!.opt.gravity[2];
    const expectedContactTime = Math.sqrt(2 * displacement / gravity);

    model!.opt.timestep = 2 ** -9;
    expect(data!.time).toEqual(0);
    while (data!.time < expectedContactTime) {
      expect(data!.ncon).toEqual(0);
      expect(data!.efc_type.length).toEqual(0);
      const prevTime = data!.time;
      mujoco.mj_step(model!, data!);
      expect(data!.time).toEqual(prevTime + model!.opt.timestep);
    }
    mujoco.mj_forward(model!, data!);
    const contact = data!.contact;
    expect(data!.ncon).toEqual(4);
    expect(data!.efc_type.length).toEqual(16);

    expectArraysClose(
        contact.get(0)!.pos.slice(0, 2), new Float64Array([-0.1, -0.1]));
    expectArraysClose(
        contact.get(1)!.pos.slice(0, 2), new Float64Array([0.1, -0.1]));
    expectArraysClose(
        contact.get(2)!.pos.slice(0, 2), new Float64Array([-0.1, 0.1]));
    expectArraysClose(
        contact.get(3)!.pos.slice(0, 2), new Float64Array([0.1, 0.1]));

    mujoco.mj_resetData(model!, data!);
    expect(data!.ncon).toEqual(0);
    expect(data!.efc_type.length).toEqual(0);
  });

  // Corresponds to bindings_test.py:test_mj_struct_equality_array
  it('should check MjContact equality with array', () => {
    const contact1 = new mujoco.MjContact();
    const contact2 = new mujoco.MjContact();
    try {
      contact1.H[3] = 1;
      expect(contact1.H).not.toEqual(contact2.H);
      contact2.H[3] = 1;
      expect(contact1).toEqual(contact2);
    } finally {
      contact1.delete();
      contact2.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mj_struct_list_equality
  it('should check MjContactVec equality', () => {
    const tempXmlFilename2 = '/tmp/model2.xml';
    writeXMLFile(tempXmlFilename2, TEST_XML);
    const model2 = mujoco.MjModel!.mj_loadXML(tempXmlFilename2);
    const data2 = new mujoco.MjData(model2);
    try {
      mujoco.mj_forward(model!, data!);
      expect(data!.ncon).toEqual(4);
      mujoco.mj_forward(model2, data2);
      expect(data2.ncon).toEqual(4);
      expect(data2.contact).toEqual(data!.contact);

      data!.qpos[3] = Math.cos(Math.PI / 8);
      data!.qpos[4] = Math.sin(Math.PI / 8);
      data!.qpos[5] = 0;
      data!.qpos[6] = 0;
      data!.qpos[2] = (Math.sqrt(2) - 1) * 0.1 - 1e-6;
      mujoco.mj_forward(model!, data!);

      expect(data!.ncon).toEqual(2);
      expect(data2.contact.size()).not.toEqual(data!.contact.size());
      expect(data!.contact).not.toBe(data!.warning);
    } finally {
      model2.delete();
      data2.delete();
      unlinkXMLFile(tempXmlFilename2);
    }
  });

  // Corresponds to bindings_test.py:test_getsetstate
  it('should get and set the state', () => {
    mujoco.mj_step(model!, data!);

    const invalidSig = 2 ** mujoco.mjtState.mjNSTATE.value;
    expect(() => {
      mujoco.mj_stateSize(model!, invalidSig);
    })
        .toThrowError(
            'MuJoCo Error: mj_stateSize: invalid state signature 32768 >= 2^mjNSTATE');

    const sig = mujoco.mjtState.mjSTATE_INTEGRATION.value;
    const size = mujoco.mj_stateSize(model!, sig);
    const stateBadSize = mujoco.DoubleBuffer.FromArray([size]);
    expect(() => {
      mujoco.mj_getState(model!, data!, stateBadSize, sig);
    })
        .toThrowError(
            'MuJoCo Error: [mj_getState] state must have size 81, got 1');

    const state0 = new mujoco.DoubleBuffer(size);
    mujoco.mj_getState(model!, data!, state0, sig);

    mujoco.mj_step(model!, data!);
    const state1a = mujoco.DoubleBuffer.FromArray(new Array(size).fill(1));
    mujoco.mj_getState(model!, data!, state1a, sig);

    mujoco.mj_setState(model!, data!, state0.GetView(), sig);
    mujoco.mj_step(model!, data!);
    const state1b = mujoco.DoubleBuffer.FromArray(new Array(size).fill(2));
    mujoco.mj_getState(model!, data!, state1b, sig);

    expectArraysEqual(state1a.GetView(), state1b.GetView());
  });

  // Corresponds to bindings_test.py:test_mj_setKeyframe
  it('should set and reset a keyframe', () => {
    mujoco.mj_step(model!, data!);

    const invalidKey = 2;
    expect(() => {
      mujoco.mj_setKeyframe(model!, data!, invalidKey);
    })
        .toThrowError(
            'MuJoCo Error: mj_setKeyframe: index must be smaller than 2 (keyframes allocated in model)');

    const validKey = 1;
    const time = data!.time;
    const qpos = new Float64Array(data!.qpos);
    const qvel = new Float64Array(data!.qvel);
    const act = new Float64Array(data!.act);
    mujoco.mj_setKeyframe(model!, data!, validKey);

    mujoco.mj_step(model!, data!);
    expect(time).not.toEqual(data!.time);

    mujoco.mj_resetDataKeyframe(model!, data!, validKey);
    expect(time).toEqual(data!.time);
    expectArraysEqual(qpos, data!.qpos);
    expectArraysEqual(qvel, data!.qvel);
    expectArraysEqual(act, data!.act);
  });

  // Corresponds to bindings_test.py:test_mj_angmomMat
  it('should compute angular momentum matrix', () => {
    data!.qvel.fill(1);
    mujoco.mj_forward(model!, data!);
    mujoco.mj_subtreeVel(model!, data!);

    const mat = new mujoco.DoubleBuffer(3 * model!.nv);
    try {
      mujoco.mj_angmomMat(model!, data!, mat, 0);

      const qvel = new Float64Array(data!.qvel);
      const subtreeAngmom = new Float64Array(data!.subtree_angmom.slice(0, 3));
      const matView = mat.GetView();
      const result = new Float64Array(3).fill(0);

      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < model!.nv; j++) {
          result[i] += matView[i * model!.nv + j] * qvel[j];
        }
      }
      expectArraysClose(result, subtreeAngmom);
    } finally {
      mat.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mj_jacSite
  it('should compute site jacobian', () => {
    mujoco.mj_forward(model!, data!);
    const siteId =
        mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_SITE.value, 'mysite');
    const jacp = new mujoco.DoubleBuffer(3 * model!.nv);
    const jacr = new mujoco.DoubleBuffer(3 * model!.nv);

    try {
      mujoco.mj_jacSite(model!, data!, jacp, null, siteId);
      const expectedJacp = new Float64Array(3 * model!.nv).fill(0);
      expectedJacp[6] = -1;
      expectArraysClose(jacp.GetView(), expectedJacp);

      mujoco.mj_jacSite(model!, data!, null, jacr, siteId);
      const expectedJacr = new Float64Array(3 * model!.nv).fill(0);
      expectedJacr[1 * model!.nv + 6] = 1;
      expectArraysClose(jacr.GetView(), expectedJacr, 2);

      jacp.GetView().fill(0);
      jacr.GetView().fill(0);
      mujoco.mj_jacSite(model!, data!, jacp, jacr, siteId);
      expectArraysClose(jacp.GetView(), expectedJacp, 2);
      expectArraysClose(jacr.GetView(), expectedJacr, 2);

      const badJacp = new mujoco.DoubleBuffer(3 * 6);
      try {
        expect(() => {
          mujoco.mj_jacSite(model!, data!, badJacp, null, siteId);
        })
            .toThrowError(
                'MuJoCo Error: [mj_jacSite] jacp must have size 30, got 18');
      } finally {
        badJacp.delete();
      }

      const badJacr = new mujoco.DoubleBuffer(4 * 7);
      try {
        expect(() => {
          mujoco.mj_jacSite(model!, data!, null, badJacr, siteId);
        })
            .toThrowError(
                'MuJoCo Error: [mj_jacSite] jacr must have size 30, got 28');
      } finally {
        badJacr.delete();
      }
    } finally {
      jacp.delete();
      jacr.delete();
    }
  });

  // Corresponds to bindings_test.py:test_can_initialize_mjv_structs
  it('should initialize mjv structs', () => {
    expect(new mujoco.MjvScene()).toBeDefined();
    expect(new mujoco.MjvCamera()).toBeDefined();
    expect(new mujoco.MjvGLCamera()).toBeDefined();
    expect(new mujoco.MjvGeom()).toBeDefined();
    expect(new mujoco.MjvLight()).toBeDefined();
    expect(new mujoco.MjvOption()).toBeDefined();
    expect(new mujoco.MjvFigure()).toBeDefined();
    expect(new mujoco.MjvScene(model, 100)).toBeDefined();
  });

  // Corresponds to bindings_test.py:test_mjv_camera
  it('should handle MjvCamera correctly', () => {
    const camera = new mujoco.MjvCamera();
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value;
    camera.fixedcamid = 2 ** 31 - 1;
    expect(camera.fixedcamid).toEqual(2 ** 31 - 1);
  });

  // Corresponds to bindings_test.py:test_mjv_scene
  it('should handle MjvScene correctly', () => {
    const scene = new mujoco.MjvScene(model, 100);
    expect(scene.ngeom).toEqual(0);
    expect(scene.maxgeom).toEqual(100);
    expect(scene.geoms.size()).toEqual(0);

    mujoco.mj_forward(model!, data!);
    mujoco.mjv_updateScene(
        model!, data!, new mujoco.MjvOption(), new mujoco.MjvPerturb(),
        new mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL.value, scene);
    expect(scene.geoms.size()).toEqual(scene.ngeom);
    expect(scene.ngeom).toBeGreaterThan(0);
  });

  // Corresponds to bindings_test.py:test_mjv_scene_without_model
  it('should initialize MjvScene without a model', () => {
    const scene = new mujoco.MjvScene();
    expect(scene.scale).toEqual(1.0);
    expect(scene.maxgeom).toEqual(0);
  });

  // Corresponds to bindings_test.py:test_inverse_fd_none
  it('should compute inverse dynamics derivatives with null outputs', () => {
    const eps = 1e-6;
    const flg_centered = false;
    expect(
        () => mujoco.mjd_inverseFD(
            model!, data!, eps, flg_centered, null, null, null, null, null,
            null, null))
        .not.toThrow();
  });

  // Corresponds to bindings_test.py:test_inverse_fd
  it('should compute inverse dynamics derivatives', () => {
    const eps = 1e-6;
    const flg_centered = false;

    const nv = model!.nv;
    const nsensordata = model!.nsensordata;
    const nC = model!.nC;

    const dfDq = new mujoco.DoubleBuffer(nv * nv);
    const dfDv = new mujoco.DoubleBuffer(nv * nv);
    const dfDa = new mujoco.DoubleBuffer(nv * nv);
    const dsDq = new mujoco.DoubleBuffer(nv * nsensordata);
    const dsDv = new mujoco.DoubleBuffer(nv * nsensordata);
    const dsDa = new mujoco.DoubleBuffer(nv * nsensordata);
    const dmDq = new mujoco.DoubleBuffer(nv * nC);

    try {
      mujoco.mjd_inverseFD(
          model!, data!, eps, flg_centered, dfDq, dfDv, dfDa, dsDq, dsDv, dsDa,
          dmDq);

      expect(norm(dfDq.GetView())).toBeGreaterThan(eps);
      expect(norm(dfDv.GetView())).toBeGreaterThan(eps);
      expect(norm(dfDa.GetView())).toBeGreaterThan(eps);
      expect(norm(dsDq.GetView())).toBeGreaterThan(eps);
      expect(norm(dsDv.GetView())).toBeGreaterThan(eps);
      expect(norm(dsDa.GetView())).toBeGreaterThan(eps);
    } finally {
      dfDq.delete();
      dfDv.delete();
      dfDa.delete();
      dsDq.delete();
      dsDv.delete();
      dsDa.delete();
      dmDq.delete();
    }
  });

  // Corresponds to bindings_test.py:test_geom_distance
  it('should compute geom distance', () => {
    mujoco.mj_forward(model!, data!);
    const fromto = new mujoco.DoubleBuffer(6);
    try {
      const dist = mujoco.mj_geomDistance(model!, data!, 0, 2, 200, fromto);
      expect(dist).toEqual(41.9);
      expectArraysClose(
          fromto.GetView(),
          new Float64Array([42.0, 0.0, 0.0, 42.0, 0.0, 41.9]));
    } finally {
      fromto.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mjd_sub_quat
  it('should compute sub quaternion derivatives', () => {
    const quat1 = [0.2, 0.3, 0.3, 0.4];
    const quat2 = [0.1, 0.2, 0.4, 0.5];
    const d1 = new mujoco.DoubleBuffer(9);
    const d2 = new mujoco.DoubleBuffer(9);
    const d3 = new mujoco.DoubleBuffer(9);
    const d4 = new mujoco.DoubleBuffer(9);
    try {
      mujoco.mjd_subQuat(quat1, quat2, d1, d2);
      mujoco.mjd_subQuat(quat1, quat2, null, d3);
      mujoco.mjd_subQuat(quat1, quat2, d4, null);
      expectArraysEqual(d2.GetView(), d3.GetView());
      expectArraysEqual(d1.GetView(), d4.GetView());
    } finally {
      d1.delete();
      d2.delete();
      d3.delete();
      d4.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mjd_quat_integrate
  it('should compute quaternion derivatives for integration', () => {
    const scale = 0.1;
    const vel = [0.2, 0.3, 0.3];
    const dQuat = new mujoco.DoubleBuffer(9);
    const dVel = new mujoco.DoubleBuffer(9);
    const dH = new mujoco.DoubleBuffer(3);
    try {
      mujoco.mjd_quatIntegrate(vel, scale, dQuat, dVel, dH);
      expect(norm(dQuat.GetView())).toBeGreaterThan(0);
      expect(norm(dVel.GetView())).toBeGreaterThan(0);
      expect(norm(dH.GetView())).toBeGreaterThan(0);
    } finally {
      dQuat.delete();
      dVel.delete();
      dH.delete();
    }
  });

  // Corresponds to bindings_test.py:test_banded
  it('should handle banded matrices', () => {
    const nTotal = 4;
    const nBand = 1;
    const nDense = 1;
    const dense = [
      1.0,
      0,
      0,
      0.1,
      0,
      2.0,
      0,
      0.2,
      0,
      0,
      3.0,
      0.3,
      0.1,
      0.2,
      0.3,
      4.0,
    ];
    const band =
        new mujoco.DoubleBuffer(nBand * (nTotal - nDense) + nDense * nTotal);
    const vec = mujoco.DoubleBuffer.FromArray([2.0, 2.0, 3.0, 4.0]);
    const res = new mujoco.DoubleBuffer(4);
    try {
      mujoco.mju_dense2Band(band, dense, nTotal, nBand, nDense);
      for (let i = 0; i < 4; i++) {
        const index = mujoco.mju_bandDiag(i, nTotal, nBand, nDense);
        expect(band.GetView()[index]).toEqual(i + 1);
      }
      const dense2 = new mujoco.DoubleBuffer(nTotal * nTotal);
      const flgSym = true;
      mujoco.mju_band2Dense(
          dense2, band.GetView(), nTotal, nBand, nDense, flgSym);
      expectArraysEqual(new Float64Array(dense), dense2.GetView());

      const nVec = 1;
      mujoco.mju_bandMulMatVec(
          res, band.GetView(), vec.GetView(), nTotal, nBand, nDense, nVec,
          flgSym);

      const expected = new Float64Array([2.4, 4.8, 10.2, 17.5]);
      expectArraysClose(res.GetView(), expected);

      const diagAdd = 0;
      const diagMul = 0;
      mujoco.mju_cholFactorBand(band, nTotal, nBand, nDense, diagAdd, diagMul);
      mujoco.mju_cholSolveBand(
          res, band.GetView(), vec.GetView(), nTotal, nBand, nDense);

      const expectedSolved = new Float64Array([1.9111, 0.9111, 0.9111, 0.8333]);
      expectArraysClose(res.GetView(), expectedSolved);
    } finally {
      band.delete();
      vec.delete();
      res.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_box_qp
  it('should handle box QP solver', () => {
    const n = 5;
    const res = new mujoco.DoubleBuffer(n);
    const r = new mujoco.DoubleBuffer(n * (n + 7));
    const index = new mujoco.IntBuffer(n);
    const h = new mujoco.DoubleBuffer(n * n);
    const g = mujoco.DoubleBuffer.FromArray(new Array(n).fill(1));
    const lower = new Array(n).fill(-1);
    const upper = new Array(n).fill(1);
    try {
      for (let i = 0; i < n; i++) {
        h.GetView()[i * (n + 1)] = 1;
      }
      const rank = mujoco.mju_boxQP(
          res, r, index, h.GetView(), g.GetView(), lower, upper);
      expect(rank).toBeGreaterThan(-1);
    } finally {
      res.delete();
      r.delete();
      index.delete();
      h.delete();
      g.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_fill
  it('should fill an array with a value', () => {
    const res = new mujoco.DoubleBuffer(3);
    try {
      mujoco.mju_fill(res, 1.5);
      expectArraysEqual(res.GetView(), new Float64Array([1.5, 1.5, 1.5]));
    } finally {
      res.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_eye
  it('should create an identity matrix', () => {
    const eye3 = new mujoco.DoubleBuffer(3 * 3);
    try {
      mujoco.mju_eye(eye3);
      const expected = new Float64Array([
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
      ]);
      expectArraysEqual(eye3.GetView(), expected);
    } finally {
      eye3.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_symmetrize
  it('should symmetrize a matrix', () => {
    const mat = [
      0, 0.066, 0.13, 0.2, 0.26, 0.33, 0.4, 0.46, 0.53, 0.6, 0.66, 0.73, 0.8,
      0.86, 0.93, 1
    ];
    const res = new mujoco.DoubleBuffer(16);
    try {
      mujoco.mju_symmetrize(res, mat, 4);
      const expected = new Float64Array([
        0, 0.163, 0.33, 0.5, 0.163, 0.33, 0.5, 0.66, 0.33, 0.5, 0.66, 0.83, 0.5,
        0.66, 0.8, 1
      ]);
      expectArraysClose(res.GetView(), expected);
    } finally {
      res.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_clip
  it('should clip a value', () => {
    expect(mujoco.mju_clip(1.5, 1.0, 2.0)).toEqual(1.5);
    expect(mujoco.mju_clip(1.5, 2.0, 3.0)).toEqual(2.0);
    expect(mujoco.mju_clip(1.5, 0.0, 1.0)).toEqual(1.0);
  });

  // Corresponds to bindings_test.py:test_mju_mul_vec_mat_vec
  it('should multiply a vector by a matrix and a vector', () => {
    const vec1 = [1.0, 2.0, 3.0];
    const vec2 = [3.0, 2.0, 1.0];
    const mat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    expect(mujoco.mju_mulVecMatVec(vec1, mat, vec2)).toEqual(204.0);
  });

  // Corresponds to bindings_test.py:test_mju_dense_to_sparse
  it('should convert a dense matrix to a sparse matrix', () => {
    const mat = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    const res = new mujoco.DoubleBuffer(3);
    const rowNnz = new mujoco.IntBuffer(2);
    const rowAdr = new mujoco.IntBuffer(2);
    const colInd = new mujoco.IntBuffer(3);
    try {
      const status =
          mujoco.mju_dense2sparse(res, mat, 2, 3, rowNnz, rowAdr, colInd);
      expect(status).toEqual(0);
      expectArraysEqual(res.GetView(), new Float64Array([1.0, 2.0, 3.0]));
      expectArraysEqual(rowNnz.GetView(), new Int32Array([1, 2]));
      expectArraysEqual(rowAdr.GetView(), new Int32Array([0, 1]));
      expectArraysEqual(colInd.GetView(), new Int32Array([1, 0, 2]));
    } finally {
      rowNnz.delete();
      rowAdr.delete();
      colInd.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_sparse_to_dense
  it('should convert a sparse matrix to a dense matrix', () => {
    const mat = [1.0, 2.0, 3.0];
    const expected = new Float64Array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    const rowNnz = [1, 2];
    const rowAdr = [0, 1];
    const colInd = [1, 0, 2];
    const res = new mujoco.DoubleBuffer(6);
    try {
      mujoco.mju_sparse2dense(res, mat, 2, 3, rowNnz, rowAdr, colInd);
      expectArraysEqual(res.GetView(), expected);
    } finally {
      res.delete();
    }
  });

  // Corresponds to bindings_test.py:test_mju_euler_to_quat
  it('should convert euler to quaternion', () => {
    const quat = new mujoco.DoubleBuffer(4);
    const euler = [0, Math.PI / 2, 0];
    const seq = 'xyz';
    try {
      mujoco.mju_euler2Quat(quat, euler, seq);
      const expectedQuat = [Math.sqrt(0.5), 0, Math.sqrt(0.5), 0.0];
      expectArraysClose(quat.GetView(), new Float64Array(expectedQuat));

      expect(() => {
        mujoco.mju_euler2Quat(quat, euler, 'xy');
      })
          .toThrowError(
              'MuJoCo Error: mju_euler2Quat: seq must contain exactly 3 characters');
      expect(() => {
        mujoco.mju_euler2Quat(quat, euler, 'xyzy');
      })
          .toThrowError(
              'MuJoCo Error: mju_euler2Quat: seq must contain exactly 3 characters');
      expect(() => {
        mujoco.mju_euler2Quat(quat, euler, 'xYp');
      })
          .toThrowError(
              `MuJoCo Error: mju_euler2Quat: seq[2] is 'p', should be one of x, y, z, X, Y, Z`);
    } finally {
      quat.delete();
    }
  });

  // Corresponds to bindings_test.py:test_texture_size
  it('should load a texture from a model', () => {
    const texFilename = 'tex.png';
    writeXMLFile(texFilename, 'tex');
    const tempXmlFilename = '/tmp/with_texture.xml';
    const TEST_XML_TEXTURE = `
      <mujoco>
        <asset>
          <texture name="tex" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
            width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
          <material name="mat" reflectance="0.3" texture="tex" texrepeat="1 1" texuniform="true"/>
        </asset>
        <worldbody>
          <geom type="plane" size="1 1 1" material="mat"/>
        </worldbody>
      </mujoco>
    `;
    writeXMLFile(tempXmlFilename, TEST_XML_TEXTURE);

    const model = mujoco.MjModel!.mj_loadXML(tempXmlFilename);
    try {
      expect(model).toBeDefined();
      expect(model!.tex_height).toEqual(new Int32Array([512]));
      expect(model!.tex_width).toEqual(new Int32Array([512]));
      // tex_adr is mjtSize* (int64_t); must surface as Int32Array, not
      // BigInt64Array, so that arithmetic like tex_adr[0] * 3 works without
      // "can't convert BigInt to number" (fixes #3496).
      expect(model!.tex_adr).toEqual(new Int32Array([0]));
      expect(() => model!.tex_adr[0] * 3).not.toThrow();
    } finally {
      model?.delete();
      unlinkXMLFile(texFilename);
      unlinkXMLFile(tempXmlFilename);
    }
  });

  it('should create distinct MjModel instances and copy correctly', () => {
    const model1 = model!;
    const model2 = new mujoco.MjModel(model1);

    try {
      assertExists(model1);
      assertExists(model2);
      expect(model1).not.toBe(model2);
      expect(model1.opt.timestep).toEqual(model2.opt.timestep);
      expect(model1.stat.meanmass).toEqual(model2.stat.meanmass);
      model1.opt.timestep = 0.123;
      expect(model1.opt.timestep).toEqual(0.123);
      expect(model2.opt.timestep).not.toEqual(model1.opt.timestep);
      expect(model2.opt.timestep).toEqual(0.002);
    } finally {
      model2.delete();
    }
  });

  it('should create distinct MjData instances and copy correctly', () => {
    const data1 = new mujoco.MjData(model!);
    const data2 = new mujoco.MjData(model!, data1);
    try {
      assertExists(data1);
      assertExists(data2);
      expect(data1).not.toBe(data2);
      expectArraysEqual(data1.qpos, data2.qpos);
      expectArraysEqual(data1.qvel, data2.qvel);
      data1.qpos[0] = 1.0;
      expect(data1.qpos[0]).toEqual(1.0);
      expect(data2.qpos[0]).not.toEqual(data1.qpos[0]);
    } finally {
      data1.delete();
      data2.delete();
    }
  });

  // Corresponds to specs_test.py:test_address
  it('should create distinct MjSpec instances and copy correctly', () => {
    const spec1 = mujoco.parseXMLString(TEST_XML);
    const spec2 = new mujoco.MjSpec(spec1);

    try {
      assertExists(spec1);
      assertExists(spec2);

      expect(spec1).not.toBe(spec2);

      expect(spec1.modelname).toEqual(spec2.modelname);
      expect(spec1.option.timestep).toEqual(spec2.option.timestep);
      expect(spec1.visual.quality.shadowsize)
          .toEqual(spec2.visual.quality.shadowsize);
      expect(spec1.stat.meanmass).toEqual(spec2.stat.meanmass);

      spec1.modelname = 'modified';
      expect(spec2.modelname).not.toEqual(spec1.modelname);
      expect(spec2.modelname).toEqual('test');
    } finally {
      spec1?.delete();
      spec2?.delete();
    }
  });

  // Corresponds to partial of user_api_test.cc:TEST_F(PluginTest, AttachPlugin)
  it('should correctly copy MjSpec instances when attaching plugins', () => {
    const xmlPlugin1 = `
      <mujoco model="MuJoCo Model">
        <worldbody>
          <body name="body"/>
        </worldbody>
      </mujoco>`;

    const spec1 = mujoco.parseXMLString(xmlPlugin1);
    const spec2 = new mujoco.MjSpec(spec1);
    const spec3 = new mujoco.MjSpec(spec1);

    try {
      assertExists(spec1);
      assertExists(spec2);
      assertExists(spec3);

      spec2.modelname = 'first_copy';
      spec3.modelname = 'second_copy';
      expect(spec1.modelname).toEqual('MuJoCo Model');
      expect(spec2.modelname).toEqual('first_copy');
      expect(spec3.modelname).toEqual('second_copy');
    } finally {
      spec1?.delete();
      spec2?.delete();
      spec3?.delete();
    }
  });

  it('should save the model to an XML file', () => {
    const tempXmlFilename = '/tmp/saved_model.xml';
    const xml = `<mujoco model="MuJoCo Model">
  <compiler angle="radian"/>
  <asset>
    <texture type="2d" colorspace="auto" name="tex" builtin="checker" mark="cross" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="512" height="512"/>
    <material name="mat" texture="tex" texuniform="true" reflectance="0.3"/>
  </asset>
  <worldbody>
    <geom size="1 1 1" type="plane" material="mat"/>
  </worldbody>
</mujoco>`;
    writeXMLFile(tempXmlFilename, xml);

    const model = mujoco.MjModel!.mj_loadXML(tempXmlFilename);
    try {
      mujoco.mj_saveLastXML(tempXmlFilename, model!);
      const savedXmlContent =
          (mujoco as any).FS.readFile(tempXmlFilename, {encoding: 'utf8'});
      // Remove whitespaces from the saved XML content to avoid flakiness.
      expect(savedXmlContent.replace(/\s/g, ''))
          .toEqual(xml.replace(/\s/g, ''));
    } finally {
      unlinkXMLFile(tempXmlFilename);
    }
  });

  it('can call mj_setLengthRange with actuators', () => {
    const tempXmlFilename = '/tmp/actuator_model.xml';
    const actuatorXml = `<mujoco model="actuator_test">
      <worldbody>
        <body>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="hinge1" type="hinge" axis="0 1 0"/>
        </body>
      </worldbody>
      <actuator>
        <position name="myactuator" joint="hinge1"/>
      </actuator>
    </mujoco>`;
    writeXMLFile(tempXmlFilename, actuatorXml);
    const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
    assertExists(model);
    const data = new mujoco.MjData(model);
    assertExists(data);
    const opt = new mujoco.MjLROpt();

    try {
      const result = mujoco.mj_setLengthRange(
          model,
          data,
          /* index= */ 0,
          opt,
      );
      expect(result).toBe(1);
    } finally {
      opt.delete();
      model.delete();
      data.delete();
      unlinkXMLFile(tempXmlFilename);
    }
  });

  it('should compile a spec and emit warnings', () => {
    const xml = `
    <mujoco>
      <worldbody>
        <flexcomp name="test" type="grid" count="4 4 1" spacing=".2 .2 .2"
                  dim="2" radius=".1"/>
      </worldbody>
    </mujoco>`;
    let spec = null;
    let model = null;
    const warnSpy = spyOn(console, 'warn');
    try {
      spec = mujoco.parseXMLString(xml);
      expect(spec).not.toBeNull();

      model = mujoco.mj_compile(spec);
      expect(model).not.toBeNull();
      expect(warnSpy).toHaveBeenCalled();
      const warningCall = warnSpy.calls.mostRecent();
      expect(warningCall.args[0]).toContain("no equality constraints or passive forces");
    } finally {
      if (spec) {
        spec.delete();
      }
      if (model) {
        model.delete();
      }
    }
  });

  it('should compile a spec from XML string with no assets', () => {
    let spec = null;
    let model = null;
    try {
      spec = mujoco.parseXMLString(TEST_XML);
      expect(spec).not.toBeNull();

      model = mujoco.mj_compile(spec);
      expect(model).not.toBeNull();
      expect(model.nq).toBeGreaterThan(0);

      const jointId =
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, 'myhinge');
      expect(jointId).toBeGreaterThanOrEqual(0);
    } finally {
      if (spec) {
        spec.delete();
      }
      if (model) {
        model.delete();
      }
    }
  });

  it('should compile a spec from XML with .obj asset', () => {
    const xml = `
    <mujoco>
      <asset>
        <mesh file="cube.obj"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="cube"/>
      </worldbody>
    </mujoco>`;

    const cube1 = `
    v -1 -1  1
    v  1 -1  1
    v -1  1  1
    v  1  1  1
    v -1  1 -1
    v  1  1 -1
    v -1 -1 -1
    v  1 -1 -1`;

    let spec: MjSpec|null = null;
    let model: MjModel|null = null;
    let vfs: MjVFS|null = null;
    try {
      spec = mujoco.parseXMLString(xml);
      assertExists(spec);

      vfs = new mujoco.MjVFS();
      vfs.addBuffer('cube.obj', new TextEncoder().encode(cube1));
      assertExists(vfs);

      model = mujoco.mj_compile(spec, vfs);
      assertExists(model);
      expect(model.nmesh).toBe(1);

      const meshId =
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH.value, 'cube');
      expect(meshId).toBeGreaterThanOrEqual(0);
    } finally {
      spec?.delete();
      vfs?.delete();
      model?.delete();
    }
  });

  it('should compile a spec from XML with 10 meshes and 10 textures', () => {
    const assetsCount = 10;
    let xml = `<mujoco>
      <asset>`;
    const cube1 = `
    v -1 -1  1
    v  1 -1  1
    v -1  1  1
    v  1  1  1
    v -1  1 -1
    v  1  1 -1
    v -1 -1 -1
    v  1 -1 -1`;
    const dummyTex = 'dummy texture data';

    for (let i = 0; i < assetsCount; i++) {
      xml += `<mesh name="cube${i}" file="cube${i}.obj"/>`;
      xml += `<texture name="tex${
          i}" type="2d" builtin="checker" width="16" height="16"/>`;
    }
    xml += `</asset><worldbody>`;
    for (let i = 0; i < assetsCount; i++) {
      xml += `<geom type="mesh" mesh="cube${i}"/>`;
    }
    xml += `</worldbody></mujoco>`;

    let spec: MjSpec|null = null;
    let model: MjModel|null = null;
    let vfs: MjVFS|null = null;
    try {
      spec = mujoco.parseXMLString(xml);
      assertExists(spec);

      vfs = new mujoco.MjVFS();
      assertExists(vfs);
      for (let i = 0; i < assetsCount; i++) {
        vfs.addBuffer(`cube${i}.obj`, new TextEncoder().encode(cube1));
        vfs.addBuffer(`tex${i}.txt`, new TextEncoder().encode(dummyTex));
      }

      model = mujoco.mj_compile(spec, vfs);
      assertExists(model);
      expect(model.nmesh).toBe(assetsCount);
      expect(model.ntex).toBe(assetsCount);

      for (let i = 0; i < assetsCount; i++) {
        const meshId = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MESH.value, `cube${i}`);
        expect(meshId).toBe(i);
        const tex = model.tex(`tex${i}`);
        expect(tex).toBeDefined();
        expect(tex!.name).toBe(`tex${i}`);
      }
    } finally {
      spec?.delete();
      vfs?.delete();
      model?.delete();
    }
  });

  describe('MjModel named access', () => {
    // Corresponds to
    // bindings_test.py:test_named_indexing_invalid_names_in_model
    it('should throw an error for invalid geom names in model', () => {
      expect(() => {
        model!.geom('badgeom');
      })
          .toThrowError(
              `MuJoCo Error: Invalid name 'badgeom' for geom. Valid names: ['mocap-sphere', 'mybox', 'myplane']`);
    });

    // Corresponds to
    // bindings_test.py:test_named_indexing_invalid_index_in_model
    it('should throw an error for invalid geom indices in model', () => {
      const numGeoms = model!.ngeom;
      expect(() => {
        model!.geom(numGeoms);
      })
          .toThrowError(
              `MuJoCo Error: Invalid index 3 for geom. Valid indices from 0 to 2`);

      expect(() => {
        model!.geom(-1);
      })
          .toThrowError(
              `MuJoCo Error: Invalid index -1 for geom. Valid indices from 0 to 2`);
    });

    // Corresponds to bindings_test.py:test_named_indexing_geom_size
    it('should correctly access geom size using named indexing', () => {
      const boxId =
          mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_GEOM.value, 'mybox');
      assertExists(boxId);

      // Check that named indexing returns the same object as id indexing.
      expect(model!.geom('mybox')).toEqual(model!.geom(boxId));
      expect(model!.geom('mybox')!.size).toEqual(model!.geom(boxId)!.size);
      expect(model!.geom('mybox')!.size.length).toEqual(3);

      // Test that the indexer is returning a view into the underlying struct.
      const sizeFromIndexer = model!.geom('mybox')!.size;
      const originalGeomSize = new Float64Array(model!.geom_size);

      model!.geom_size.set([7, 11, 13], boxId * 3);
      expectArraysEqual(sizeFromIndexer, new Float64Array([7, 11, 13]));

      model!.geom('mybox')!.size.set([5, 3, 2]);
      expectArraysEqual(
          model!.geom_size.slice(boxId * 3, boxId * 3 + 3),
          new Float64Array([5, 3, 2]));
    });

    // Corresponds to bindings_test.py:test_named_indexing_geom_quat
    it('should correctly access geom quat using named indexing', () => {
      const boxId =
          mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_GEOM.value, 'mybox');
      assertExists(boxId);

      // Check that named indexing returns the same object as id indexing.
      expect(model!.geom('mybox')).toEqual(model!.geom(boxId));
      expect(model!.geom('mybox')!.quat).toEqual(model!.geom(boxId)!.quat);
      expect(model!.geom('mybox')!.quat.length).toEqual(4);

      // Test that the indexer is returning a view into the underlying struct.
      const quatFromIndexer = model!.geom('mybox')!.quat;

      model!.geom_quat.set([5, 10, 15, 20], boxId * 4);
      expectArraysEqual(quatFromIndexer, new Float64Array([5, 10, 15, 20]));

      model!.geom('mybox')!.quat.set([12, 9, 6, 3]);
      expectArraysEqual(
          model!.geom_quat.slice(boxId * 4, boxId * 4 + 4),
          new Float64Array([12, 9, 6, 3]));
    });

    it('should support named access for Joints', () => {
      const xml = `<mujoco model="Box falling">
      <option viscosity="1"/>
      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="MyFloor" type="plane" size="1 1 0.1" rgba=".9 0 0 1" user="5 4 3 2 1"/>
        <geom name="MyWall" type="plane" size="0.1 1 0.1" rgba="0 1 0 1" user="5 4 3"/>
        <body pos="0 0 1" name="MyBox">
          <joint type="free" name="root"/>
          <geom name="MyBoxGeom" type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
          <body pos=".1 0 0" name="MyHingeBody">
            <joint frictionloss="0.4" type="hinge" name="hinge" axis="0 1 0"/>
            <geom type="capsule" size=".05 .05" rgba="0 0 .9 1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>`;
      const tempXmlFilename = '/tmp/jnt.xml';
      writeXMLFile(tempXmlFilename, xml);
      const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      try {
        assertExists(model);
        const j0 = model.jnt('root');
        expect(j0.id).toBe(0);
        expect(j0.name).toBe('root');
        expect(j0.type).toBe(mujoco.mjtJoint.mjJNT_FREE.value);
        expect(j0.qposadr).toBe(0);
        expect(j0.dofadr).toBe(0);
        expect(j0.axis).toEqual(new Float64Array([0, 0, 1]));
        expect(j0.type).toEqual(mujoco.mjtJoint.mjJNT_FREE.value);

        const j0qposStart = model.jnt_qposadr[j0.id];
        const j0qposEnd =
            (j0.id < model.njnt - 1) ? model.jnt_qposadr[j0.id + 1] : model.nq;
        const expectedJ0qpos = model.qpos0.slice(j0qposStart, j0qposEnd);
        expectArraysEqual(
            expectedJ0qpos, new Float64Array([0, 0, 1, 1, 0, 0, 0]));
        expectArraysEqual(j0.qpos0, expectedJ0qpos);

        const j0dofStart = model.jnt_dofadr[j0.id];
        const j0dofEnd =
            (j0.id < model.njnt - 1) ? model.jnt_dofadr[j0.id + 1] : model.nv;
        const expectedJ0dof = model.dof_bodyid.slice(j0dofStart, j0dofEnd);
        expectArraysEqual(expectedJ0dof, new Int32Array([1, 1, 1, 1, 1, 1]));
        expectArraysEqual(j0.bodyid, expectedJ0dof);


        const j1 = model.jnt('hinge');
        expect(j1.id).toBe(1);
        expect(j1.name).toBe('hinge');
        expect(j1.type).toBe(mujoco.mjtJoint.mjJNT_HINGE.value);
        expect(j1.qposadr).toBe(7);
        expect(j1.dofadr).toBe(6);
        expect(j1.axis).toEqual(new Float64Array([0, 1, 0]));
        expect(j1.frictionloss).toEqual(new Float64Array([0.4]));

        const j1qposStart = model.jnt_qposadr[j1.id];
        const j1qposEnd =
            (j1.id < model.njnt - 1) ? model.jnt_qposadr[j1.id + 1] : model.nq;
        const expectedJ1qpos = model.qpos0.slice(j1qposStart, j1qposEnd);
        expectArraysEqual(expectedJ1qpos, new Float64Array([0]));
        expectArraysEqual(j1.qpos0, expectedJ1qpos);

        const j1dofStart = model.jnt_dofadr[j1.id];
        const j1dofEnd =
            (j1.id < model.njnt - 1) ? model.jnt_dofadr[j1.id + 1] : model.nv;
        const expectedJ1dof = model.dof_bodyid.slice(j1dofStart, j1dofEnd);
        expectArraysEqual(expectedJ1dof, new Int32Array([2]));
        expectArraysEqual(j1.bodyid, new Int32Array([2]));
      } finally {
        model?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });

    it('should support named access for HFields', () => {
      const xml = `<mujoco model="test_hfield">
        <asset>
          <hfield name="hf" nrow="2" ncol="3" size="1 2 0.1 0.2"/>
        </asset>
        <worldbody>
          <geom type="hfield" hfield="hf"/>
        </worldbody>
      </mujoco>`;
      const tempXmlFilename = '/tmp/hf.xml';
      writeXMLFile(tempXmlFilename, xml);
      const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      try {
        assertExists(model);
        const hf = model.hfield('hf');
        expect(hf.id).toBe(0);
        expect(hf.name).toBe('hf');
        expect(hf.nrow).toBe(2);
        expect(hf.ncol).toBe(3);
        expectArraysClose(hf.size, new Float64Array([1, 2, 0.1, 0.2]));
        const expectedDataLength =
            model.hfield_nrow[hf.id] * model.hfield_ncol[hf.id];
        expect(expectedDataLength).toBe(6);
        expect(hf.data.length).toBe(expectedDataLength);
      } finally {
        model?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });

    it('should support named access for Numeric', () => {
      const xml = `<mujoco>
        <custom>
          <numeric name="n1" data="1 2 3"/>
          <numeric name="n2" data="4 5 6 7"/>
        </custom>
      </mujoco>`;
      const tempXmlFilename = '/tmp/num.xml';
      writeXMLFile(tempXmlFilename, xml);
      const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      try {
        assertExists(model);
        const n1 = model.numeric('n1');
        const expectedN1data = model.numeric_data.slice(
            model.numeric_adr[n1.id],
            model.numeric_adr[n1.id] + model.numeric_size[n1.id]);
        expect(n1.id).toBe(0);
        expect(n1.name).toBe('n1');
        expect(n1.size).toBe(3);
        expect(expectedN1data).toEqual(new Float64Array([1, 2, 3]));
        expectArraysClose(n1.data, expectedN1data);

        const n2 = model.numeric('n2');
        const expectedN2data = model.numeric_data.slice(
            model.numeric_adr[n2.id],
            model.numeric_adr[n2.id] + model.numeric_size[n2.id]);
        expect(n2.id).toBe(1);
        expect(n2.name).toBe('n2');
        expect(n2.size).toBe(4);
        expect(expectedN2data).toEqual(new Float64Array([4, 5, 6, 7]));
        expectArraysClose(n2.data, expectedN2data);
      } finally {
        model?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });

    it('should support named access for Texture', () => {
      const xml = `<mujoco>
        <asset>
          <texture name="t1" type="2d" builtin="checker" width="10"
          height="12"/>
        </asset>
      </mujoco>`;
      const tempXmlFilename = '/tmp/tex.xml';
      writeXMLFile(tempXmlFilename, xml);
      const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      try {
        assertExists(model);
        const t1 = model.tex('t1');
        const texAdr = Number(model.tex_adr[t1.id]);
        const expectedData = model.tex_data.slice(
            texAdr,
            texAdr +
                model.tex_height[t1.id] * model.tex_width[t1.id] *
                    model.tex_nchannel[t1.id]);

        expect(t1.id).toBe(0);
        expect(t1.name).toBe('t1');
        expect(t1.type).toBe(mujoco.mjtTexture.mjTEXTURE_2D.value);
        expect(t1.width).toBe(10);
        expect(t1.height).toBe(12);
        expect(t1.nchannel).toBe(3);
        expect(t1.data.length).toBe(t1.width * t1.height * t1.nchannel);
        expectArraysClose(t1.data, expectedData);
      } finally {
        model?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });

    it('should support named access for Tuple', () => {
      const xml = `<mujoco>
        <worldbody>
          <geom name="g1" type="sphere" size=".1"/>
          <geom name="g2" type="box" size=".1 .1 .1"/>
        </worldbody>
        <custom>
          <tuple name="tup1">
            <element objtype="geom" objname="g1" prm="0.1"/>
            <element objtype="geom" objname="g2" prm="0.2"/>
          </tuple>
        </custom>
      </mujoco>`;
      const tempXmlFilename = '/tmp/tup.xml';
      writeXMLFile(tempXmlFilename, xml);
      const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      try {
        assertExists(model);
        const t1 = model.tuple('tup1');
        const expectedObjtype = model.tuple_objtype.slice(
            model.tuple_adr[t1.id],
            model.tuple_adr[t1.id] + model.tuple_size[t1.id]);
        expect(t1.id).toBe(0);
        expect(t1.name).toBe('tup1');
        expect(t1.size).toBe(2);
        expectArraysEqual(
            expectedObjtype, new Int32Array([
              mujoco.mjtObj.mjOBJ_GEOM.value, mujoco.mjtObj.mjOBJ_GEOM.value
            ]));
        expectArraysEqual(t1.objtype, expectedObjtype);
      } finally {
        model?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });
  });

  describe('MjData named access', () => {
    it('should support named access for MjData', () => {
      mujoco.mj_forward(model!, data!);

      // Actuator
      expect(model!.nu).toBe(1);
      const actuator = data!.actuator(0);
      expect(actuator.name).toBe('myactuator');
      expect(actuator.velocity).toBe(0);
      expect(actuator.force).toBe(0);
      // the hinge's dof follows the 6 of the free joint
      expectArraysClose(
          actuator.moment, new Float64Array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]));

      // Body
      expect(model!.nbody).toBe(5);
      expect(data!.body('mybox').name).toBe('mybox');
      expectArraysClose(
          data!.body('mybox').xpos, new Float64Array([0, 0, 0.1]));
      expect(data!.body('myhinge-body').name).toBe('myhinge-body');
      expectArraysClose(
          data!.body('myhinge-body').xpos, new Float64Array([0, 0, 1]));
      expect(data!.body('myball-body').name).toBe('myball-body');
      expectArraysClose(
          data!.body('myball-body').xpos, new Float64Array([2, 0, 1]));
      expect(data!.body('mocap-body').name).toBe('mocap-body');
      expectArraysClose(
          data!.body('mocap-body').xpos, new Float64Array([42, 0, 42]));

      // Geom
      expect(model!.ngeom).toBe(3);
      expect(data!.geom('myplane').name).toBe('myplane');
      expectArraysClose(
          data!.geom('myplane').xpos, new Float64Array([0, 0, 0]));
      expectArraysClose(
          data!.geom('myplane').xmat,
          new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));

      expect(data!.geom('mybox').name).toBe('mybox');
      expectArraysClose(
          data!.geom('mybox').xpos, new Float64Array([0, 0, 0.1]));
      expectArraysClose(
          data!.geom('mybox').xmat,
          new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));

      expect(data!.geom('mocap-sphere').name).toBe('mocap-sphere');
      expectArraysClose(
          data!.geom('mocap-sphere').xpos, new Float64Array([42, 0, 42]));
      expectArraysClose(
          data!.geom('mocap-sphere').xmat,
          new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));

      // Joint
      expect(model!.njnt).toBe(3);
      const freeJnt = data!.jnt('myfree');
      expect(freeJnt.name).toBe('myfree');
      expectArraysClose(freeJnt.xanchor, new Float64Array([0, 0, 0.1]));
      expectArraysClose(freeJnt.xaxis, new Float64Array([0, 0, 1]));

      const hingeJnt = data!.jnt('myhinge');
      expect(hingeJnt.name).toBe('myhinge');
      expectArraysClose(hingeJnt.xanchor, new Float64Array([0, 0, 1]));
      expectArraysClose(hingeJnt.xaxis, new Float64Array([0, 1, 0]));

      const ballJnt = data!.jnt('myball');
      expect(ballJnt.name).toBe('myball');
      expectArraysClose(ballJnt.xanchor, new Float64Array([2, 0, 1]));
      expectArraysClose(ballJnt.xaxis, new Float64Array([0, 0, 1]));

      // Sensor
      expect(model!.nsensor).toBe(2);
      const jointvelSensor = data!.sensor('myjointvel');
      expect(jointvelSensor.name).toBe('myjointvel');
      expectArraysClose(jointvelSensor.data, new Float64Array([0]));

      const accelSensor = data!.sensor('myaccelerometer');
      expect(accelSensor.name).toBe('myaccelerometer');
      expectArraysClose(accelSensor.data, new Float64Array([0, 0, 9.81]));

      // Site
      expect(model!.nsite).toBe(1);
      const site = data!.site('mysite');
      expect(site.name).toBe('mysite');
      expectArraysClose(site.xpos, new Float64Array([0, 0, 0]));
      expectArraysClose(
          site.xmat, new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
    });

    // Corresponds to bindings_test.py:test_named_indexing_mimo_actuators
    it('should follow the controls and outputs of each actuator', () => {
      const xml = `
      <mujoco>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size=".1"/>
            <body>
              <joint name="ball" type="ball"/>
              <geom size=".1" pos=".2 0 0"/>
            </body>
          </body>
        </worldbody>
        <actuator>
          <motor name="before" joint="slide" gear="2"/>
          <orientation name="orient" joint="ball" kp="1" input="quat"/>
          <motor name="after" joint="slide" gear="3"/>
        </actuator>
      </mujoco>`;
      const tempXmlFilename = '/tmp/mimo.xml';
      writeXMLFile(tempXmlFilename, xml);
      const mimoModel = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      let mimoData: MjData|null = null;
      try {
        assertExists(mimoModel);
        mimoData = new mujoco.MjData(mimoModel);
        for (let i = 0; i < mimoModel.nu; i++) mimoData.ctrl[i] = i;
        mujoco.mj_forward(mimoModel, mimoData);
        expect(mimoModel.nactuator).toBe(3);
        expect(mimoModel.nu).toBe(6);
        expect(mimoModel.nout).toBe(5);

        // one control and output: a single value, at the actuator's address
        expect(mimoData.actuator('after').ctrl).toBe(5);
        expect(mimoData.actuator('after').force)
            .toBe(mimoData.actuator_force[4]);
        expectArraysEqual(
            mimoModel.actuator('after').gear,
            new Float64Array([3, 0, 0, 0, 0, 0]));

        // several: all of them
        expectArraysEqual(
            mimoData.actuator('orient').ctrl, new Float64Array([1, 2, 3, 4]));
        expectArraysEqual(
            mimoData.actuator('orient').force,
            mimoData.actuator_force.slice(1, 4));
        expect(mimoModel.actuator('orient').gear.length).toBe(18);

        // moment: a dense copy of the sparse rows
        const moment = mimoData.actuator('after').moment;
        expectArraysClose(moment, new Float64Array([3, 0, 0, 0]));
        expect(mimoData.actuator('orient').moment.length).toBe(3 * 4);

        // writes reach mjData
        mimoData.actuator('after').ctrl = 7;
        expect(mimoData.ctrl[5]).toBe(7);
      } finally {
        mimoData?.delete();
        mimoModel?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });

    // Corresponds to bindings_test.py:test_indexer_name_id
    it('should support named and indexed access for MjData geoms', () => {
      const xml = `
      <mujoco>
        <worldbody>
          <geom name="mygeom" size="1" pos="0 0 1"/>
          <geom size="2" pos="0 0 2"/>
          <geom size="3" pos="0 0 3"/>
          <geom name="myothergeom" size="4" pos="0 0 4"/>
          <geom size="5" pos="0 0 5"/>
        </worldbody>
      </mujoco>
      `;
      const tempXmlFilename = '/tmp/geom_idx.xml';
      writeXMLFile(tempXmlFilename, xml);
      const model = mujoco.MjModel.mj_loadXML(tempXmlFilename);
      const data = new mujoco.MjData(model);
      try {
        assertExists(model);
        assertExists(data);
        mujoco.mj_forward(model, data);

        expect(data.geom('mygeom').id).toBe(0);
        expect(data.geom('myothergeom').id).toBe(3);
        expect(data.geom(0).name).toBe('mygeom');
        expect(data.geom(1).name).toBe('');
        expect(data.geom(2).name).toBe('');
        expect(data.geom(3).name).toBe('myothergeom');
        expect(data.geom(4).name).toBe('');

        expect(data.geom(0).xpos[2]).toBeCloseTo(1);
        expect(data.geom(1).xpos[2]).toBeCloseTo(2);
        expect(data.geom(2).xpos[2]).toBeCloseTo(3);
        expect(data.geom(3).xpos[2]).toBeCloseTo(4);
        expect(data.geom(4).xpos[2]).toBeCloseTo(5);
      } finally {
        model?.delete();
        data?.delete();
        unlinkXMLFile(tempXmlFilename);
      }
    });

    // Corresponds to bindings_test.py:test_named_indexing_invalid_names_in_data
    it('should throw an error for invalid geom names in data', () => {
      expect(() => {
        data!.geom('badgeom');
      })
          .toThrowError(
              `MuJoCo Error: Invalid name 'badgeom' for geom. Valid names: ['mocap-sphere', 'mybox', 'myplane']`);
    });

    // Corresponds to bindings_test.py:test_named_indexing_invalid_index_in_data
    it('should throw an error for invalid geom indices in data', () => {
      expect(() => {
        data!.geom(3);
      })
          .toThrowError(
              'MuJoCo Error: Invalid index 3 for geom. Valid indices from 0 to 2');
      expect(() => {
        data!.geom(-1);
      })
          .toThrowError(
              'MuJoCo Error: Invalid index -1 for geom. Valid indices from 0 to 2');
    });

    // Corresponds to bindings_test.py:test_named_indexing_actuator_ctrl
    it('should support named access for Actuator ctrl', () => {
      const actuatorId = mujoco.mj_name2id(
          model!, mujoco.mjtObj.mjOBJ_ACTUATOR.value, 'myactuator');
      assertExists(actuatorId);

      const actuator = data!.actuator('myactuator');
      expect(actuator).toEqual(data!.actuator(actuatorId));
      expect(actuator.ctrl).toEqual(data!.actuator(actuatorId)!.ctrl);

      // Test that the indexer is returning a view into the underlying
      data!.ctrl[actuatorId] = 5;
      expect(actuator.ctrl).toBe(5);
      actuator.ctrl = 7;
      expect(data!.ctrl[actuatorId]).toBe(7);
    });

    // Corresponds to bindings_test.py:test_named_indexing_ragged_qpos
    it('should support named access for Joint qpos', () => {
      const balljointId =
          mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_JOINT.value, 'myball');
      assertExists(balljointId);

      const ballJnt = data!.jnt('myball');
      expect(ballJnt).toEqual(data!.jnt(balljointId));
      expect(ballJnt.qpos).toEqual(data!.jnt(balljointId)!.qpos);

      // Test that the indexer is returning a view into the underlying struct.
      const qposFromIndexer = ballJnt.qpos;
      const qposIdx = model!.jnt_qposadr[balljointId];
      data!.qpos.set([4, 5, 6, 7], qposIdx);
      expectArraysEqual(qposFromIndexer, new Float64Array([4, 5, 6, 7]));
      ballJnt.qpos.set(new Float64Array([9, 8, 7, 6]))
      expectArraysEqual(
          data!.qpos.slice(qposIdx, qposIdx + 4),
          new Float64Array([9, 8, 7, 6]));
    });

    // Corresponds to bindings_test.py:test_named_indexing_ragged2d_cdof
    it('should support named access for Joint cdof', () => {
      const freejointId =
          mujoco.mj_name2id(model!, mujoco.mjtObj.mjOBJ_JOINT.value, 'myfree');
      assertExists(freejointId);
      mujoco.mj_forward(model!, data!);

      const freeJnt = data!.jnt('myfree');
      expect(freeJnt).toEqual(data!.jnt(freejointId));
      expect(freeJnt.cdof).toEqual(data!.jnt(freejointId)!.cdof);
      expect(freeJnt.cdof.length).toBe(36);  // 6x6

      // Test that the indexer is returning a view into the underlying
      const cdofFromIndexer = freeJnt.cdof;
      const dofIdx = model!.jnt_dofadr[freejointId];
      const testArray = new Float64Array(36);
      for (let i = 0; i < 36; i++) {
        testArray[i] = i;
      }
      data!.cdof.set(testArray);
      expectArraysEqual(freeJnt.cdof, testArray);
      const expectedCdof = new Float64Array(36).fill(42);
      freeJnt.cdof.set(expectedCdof);
      expectArraysEqual(data!.cdof.slice(dofIdx, dofIdx + 36), expectedCdof);
    });
  });

  it('should save model to buffer', () => {
    assertExists(model);
    const modelSize = mujoco.mj_sizeModel(model);
    const buffer = new mujoco.Uint8Buffer(Number(modelSize));
    try {
      mujoco.mj_saveModel(model, null, buffer);
      // Test the first 3 byte values of the buffer.
      expect(buffer.GetView()[0]).toBe(49);
      expect(buffer.GetView()[1]).toBe(212);
      expect(buffer.GetView()[2]).toBe(0);
    } finally {
      buffer.delete();
    }
  });

  it('should save model to file', () => {
    assertExists(model);
    const filename = '/tmp/test.mjb';
    const modelSize = mujoco.mj_sizeModel(model);
    const buffer = new mujoco.Uint8Buffer(Number(modelSize));
    try {
      mujoco.mj_saveModel(model, filename, null);
      const fileContent =
          (mujoco as any).FS.readFile(filename, {encoding: 'binary'});

      mujoco.mj_saveModel(model, null, buffer);
      const bufferContent = buffer.GetView();

      expect(fileContent.length).toBe(bufferContent.length);
      expect(fileContent).toEqual(bufferContent);
    } finally {
      buffer.delete();
      unlinkXMLFile(filename);
    }
  });

  it('should load and save a model with assets to binary', () => {
    const xmlContent = `
    <mujoco model="test_binary_save">
      <asset>
        <mesh file="cube.obj"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="cube"/>
      </worldbody>
    </mujoco>`;
    const cube1 = `
    v -1 -1  1
    v  1 -1  1
    v -1  1  1
    v  1  1  1
    v -1  1 -1
    v  1  1 -1
    v -1 -1 -1
    v  1 -1 -1`;
    const xmlFilename = '/tmp/binary_test.xml';
    const objFilename = '/tmp/cube.obj';
    const mjbFilename = '/tmp/binary_test.mjb';

    writeXMLFile(xmlFilename, xmlContent);
    writeXMLFile(objFilename, cube1);

    let model: MjModel|null = null;
    let binaryModel: MjModel|null = null;
    let vfs: MjVFS|null = null;

    try {
      model = mujoco.MjModel.mj_loadXML(xmlFilename);
      assertExists(model);

      mujoco.mj_saveModel(model, mjbFilename, null);

      vfs = new mujoco.MjVFS();
      vfs.addBuffer(objFilename, new TextEncoder().encode(cube1));
      binaryModel = mujoco.MjModel.mj_loadModel(mjbFilename, vfs);
      assertExists(binaryModel);

      expect(mujoco.mj_sizeModel(binaryModel))
          .toEqual(mujoco.mj_sizeModel(model));
      expect(binaryModel.nbody).toEqual(model!.nbody);
      expect(binaryModel.nq).toEqual(model!.nq);
      expect(binaryModel.nv).toEqual(model!.nv);
      expect(binaryModel.njnt).toEqual(model!.njnt);
      expect(binaryModel.nmesh).toEqual(model!.nmesh);
    } finally {
      model?.delete();
      binaryModel?.delete();
      vfs?.delete();
      unlinkXMLFile(xmlFilename);
      unlinkXMLFile(objFilename);
      unlinkXMLFile(mjbFilename);
    }
  });

  // Corresponds to bindings_test.py:test_mj_saveModel
  it('should save and load a model from binary', () => {
    const mjbFilename = '/tmp/saved_model.mjb';
    let binaryModel: MjModel|null = null;
    let vfs: MjVFS|null = null;
    try {
      mujoco.mj_saveModel(model!, mjbFilename, null);
      const bufSize = mujoco.mj_sizeModel(model!);

      vfs = new mujoco.MjVFS();
      binaryModel = mujoco.MjModel.mj_loadModel(mjbFilename, vfs);
      assertExists(binaryModel);

      expect(mujoco.mj_sizeModel(binaryModel)).toEqual(bufSize);
      expect(binaryModel.nbody).toEqual(model!.nbody);
      expect(binaryModel.nq).toEqual(model!.nq);
      expect(binaryModel.nv).toEqual(model!.nv);
      expect(binaryModel.njnt).toEqual(model!.njnt);
      expect(binaryModel.nmesh).toEqual(model!.nmesh);
    } finally {
      binaryModel?.delete();
      vfs?.delete();
      unlinkXMLFile(mjbFilename);
    }
  });

  it('should load XML with assets from VFS', () => {
    const xml = `
    <mujoco>
      <asset>
        <mesh file="cube.obj"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="cube"/>
      </worldbody>
    </mujoco>`;

    const cube1 = `
    v -1 -1  1
    v  1 -1  1
    v -1  1  1
    v  1  1  1
    v -1  1 -1
    v  1  1 -1
    v -1 -1 -1
    v  1 -1 -1`;

    const xmlFilename = '/tmp/with_vfs.xml';
    writeXMLFile(xmlFilename, xml);

    let model: MjModel|null = null;
    let vfs: MjVFS|null = null;
    try {
      vfs = new mujoco.MjVFS();
      vfs.addBuffer('cube.obj', new TextEncoder().encode(cube1));
      assertExists(vfs);

      model = mujoco.MjModel.mj_loadXML(xmlFilename, vfs);
      assertExists(model);
      expect(model.nmesh).toBe(1);

      const meshId =
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH.value, 'cube');
      expect(meshId).toBeGreaterThanOrEqual(0);
    } finally {
      model?.delete();
      vfs?.delete();
      unlinkXMLFile(xmlFilename);
    }
  });

  it('should load a model from an XML string', () => {
    let model: MjModel|null = null;
    try {
      model = mujoco.from_xml_string(TEST_XML);
      assertExists(model);
      expect(model.nbody).toBe(5);
      expect(model.ngeom).toBe(3);
    } finally {
      model?.delete();
    }
  });

  it('should load a model from an XML string with VFS', () => {
    const xml = `
    <mujoco>
      <asset>
        <mesh file="cube.obj"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="cube"/>
      </worldbody>
    </mujoco>`;

    const cube1 = `
    v -1 -1  1
    v  1 -1  1
    v -1  1  1
    v  1  1  1
    v -1  1 -1
    v  1  1 -1
    v -1 -1 -1
    v  1 -1 -1`;

    let model: MjModel|null = null;
    let vfs: MjVFS|null = null;
    try {
      vfs = new mujoco.MjVFS();
      vfs.addBuffer('cube.obj', new TextEncoder().encode(cube1));
      assertExists(vfs);

      model = mujoco.from_xml_string(xml, vfs);
      assertExists(model);
      expect(model.nmesh).toBe(1);

      const meshId =
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH.value, 'cube');
      expect(meshId).toBeGreaterThanOrEqual(0);
    } finally {
      model?.delete();
      vfs?.delete();
    }
  });
});
