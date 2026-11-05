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

(function() {
  const cBlue = (s) => `<span style="color:#3b82f6;font-weight:600;">${s}</span>`;
  const cPurple = (s) => `<span style="color:#8b5cf6;font-weight:600;">${s}</span>`;
  const cGreen = (s) => `<span style="color:#10b981;font-weight:600;">${s}</span>`;
  const cAmber = (s) => `<span style="color:#d97706;font-weight:600;">${s}</span>`;

  function rot(x, y, th) {
    const c = Math.cos(th), s = Math.sin(th);
    return [c * x - s * y, s * x + c * y];
  }
  function boxVerts(cx, cy, hx, hy, th) {
    return [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([x, y]) => {
      const [rx, ry] = rot(x, y, th);
      return [cx + rx, cy + ry];
    });
  }
  function ptSegClosest(px, py, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay;
    const t = Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)));
    const qx = ax + t * abx, qy = ay + t * aby;
    return { d: Math.hypot(px - qx, py - qy), q: [qx, qy] };
  }
  function polyClosest(v1, v2) {
    let best = { d: 1e9, p1: [0, 0], p2: [0, 0] };
    for (const p1 of v1) {
      for (let i = 0; i < 4; i++) {
        const a = v2[i], b = v2[(i + 1) % 4];
        const res = ptSegClosest(p1[0], p1[1], a[0], a[1], b[0], b[1]);
        if (res.d < best.d) best = { d: res.d, p1: p1, p2: res.q };
      }
    }
    for (const p2 of v2) {
      for (let i = 0; i < 4; i++) {
        const a = v1[i], b = v1[(i + 1) % 4];
        const res = ptSegClosest(p2[0], p2[1], a[0], a[1], b[0], b[1]);
        if (res.d < best.d) best = { d: res.d, p1: res.q, p2: p2 };
      }
    }
    const nx = (best.p2[0] - best.p1[0]) / best.d;
    const ny = (best.p2[1] - best.p1[1]) / best.d;
    return { ...best, n: [nx, ny] };
  }
  function ptsStr(verts) {
    return verts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  }
  function arcPath(cx, cy, radius, a1, a2, ccw) {
    const x1 = cx + radius * Math.cos(a1), y1 = cy + radius * Math.sin(a1);
    const x2 = cx + radius * Math.cos(a2), y2 = cy + radius * Math.sin(a2);
    const sweep = ccw ? 0 : 1;
    return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${radius} ${radius} 0 0 ${sweep} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  }

  // Scene 0: Basic example (Sphere gliding across a box corner onto its top face, dw = 0)
  function buildBasicScene() {
    const c0 = [135, 68], dx = [380, 118], r = 32;
    const box = { xmin: 230, xmax: 520, ymin: 162, ymax: 226 };
    const d_target = 3.0;
    const steps = [];
    let t = 0;
    for (let k = 0; k <= 2; k++) {
      const c1 = [c0[0] + t * dx[0], c0[1] + t * dx[1]];
      const bx = Math.max(box.xmin, Math.min(box.xmax, c1[0]));
      const by = Math.max(box.ymin, Math.min(box.ymax, c1[1]));
      const dc = Math.hypot(bx - c1[0], by - c1[1]);
      const n = [(bx - c1[0]) / dc, (by - c1[1]) / dc];
      const p1 = [c1[0] + r * n[0], c1[1] + r * n[1]];
      const p2 = [bx, by];
      const d = dc - r;
      const v_bound = n[0] * dx[0] + n[1] * dx[1];
      const dt = k < 2 ? Math.max(0, (d - d_target) / v_bound) : 0;
      steps.push({ k, t, c1, r, box, p1, p2, n, d, v_bound, dt, t_next: t + dt, c0, dx });
      t += dt;
    }
    return {
      steps,
      renderStep(s, idx) {
        const v1End = [s.c1[0] + s.dx[0] * 0.20, s.c1[1] + s.dx[1] * 0.20];
        const mid = [(s.p1[0] + s.p2[0]) / 2, (s.p1[1] + s.p2[1]) / 2];
        const tx = -s.n[1], ty = s.n[0];
        const planeHalf = 85;
        const pl1 = [s.p2[0] - tx * planeHalf, s.p2[1] - ty * planeHalf];
        const pl2 = [s.p2[0] + tx * planeHalf, s.p2[1] + ty * planeHalf];
        const nBase = [s.p2[0] - tx * 52, s.p2[1] - ty * 52];
        const nEnd = [nBase[0] + s.n[0] * 24, nBase[1] + s.n[1] * 24];

        const svg = `
          <!-- Swept center trajectory t in [0, 1] -->
          <line x1="${s.c0[0]}" y1="${s.c0[1]}" x2="${s.c0[0] + s.dx[0]}" y2="${s.c0[1] + s.dx[1]}" class="ca-path1" stroke-width="0.9" stroke-dasharray="3 3"/>

          <!-- Static Box (geom 2) -->
          <rect x="${s.box.xmin}" y="${s.box.ymin}" width="${s.box.xmax - s.box.xmin}" height="${s.box.ymax - s.box.ymin}" class="ca-g2-fill" stroke-width="1.2"/>
          <text x="${(s.box.xmin + s.box.xmax) / 2}" y="${(s.box.ymin + s.box.ymax) / 2 + 4}" text-anchor="middle" class="ca-txt" font-size="11" font-weight="600">static box (geom 2)</text>

          <!-- Moving Sphere (geom 1) -->
          <circle cx="${s.c1[0].toFixed(1)}" cy="${s.c1[1].toFixed(1)}" r="${s.r}" class="ca-g1-fill" stroke-width="1.2"/>
          <circle cx="${s.c1[0].toFixed(1)}" cy="${s.c1[1].toFixed(1)}" r="2.2" fill="#2563eb"/>
          <line x1="${s.c1[0].toFixed(1)}" y1="${s.c1[1].toFixed(1)}" x2="${v1End[0].toFixed(1)}" y2="${v1End[1].toFixed(1)}" stroke="#3b82f6" stroke-width="1.15" marker-end="url(#ca-arr-b)"/>
          <text x="${(s.c1[0] - 28).toFixed(1)}" y="${(s.c1[1] - s.r - 8).toFixed(1)}" class="ca-txt" font-size="11" font-weight="600">sphere (t=${s.t.toFixed(2)})</text>

          <!-- SAT separating plane perpendicular to n -->
          <line x1="${pl1[0].toFixed(1)}" y1="${pl1[1].toFixed(1)}" x2="${pl2[0].toFixed(1)}" y2="${pl2[1].toFixed(1)}" class="ca-plane" stroke-width="1.0" stroke-dasharray="4 3"/>
          <line x1="${nBase[0].toFixed(1)}" y1="${nBase[1].toFixed(1)}" x2="${nEnd[0].toFixed(1)}" y2="${nEnd[1].toFixed(1)}" stroke="#10b981" stroke-width="1.25" marker-end="url(#ca-arr-g)"/>
          <text x="${(nEnd[0] + 4).toFixed(1)}" y="${(nEnd[1] + 4).toFixed(1)}" fill="#10b981" font-size="11" font-weight="700">n</text>

          <!-- GJK closest segment dist(t) -->
          <line x1="${s.p1[0].toFixed(1)}" y1="${s.p1[1].toFixed(1)}" x2="${s.p2[0].toFixed(1)}" y2="${s.p2[1].toFixed(1)}" stroke="#f59e0b" stroke-width="1.1" stroke-dasharray="3 2"/>
          <circle cx="${s.p1[0].toFixed(1)}" cy="${s.p1[1].toFixed(1)}" r="2.5" fill="#2563eb"/>
          <circle cx="${s.p2[0].toFixed(1)}" cy="${s.p2[1].toFixed(1)}" r="2.5" fill="#7c3aed"/>
          <text x="${(mid[0] + tx * 28).toFixed(1)}" y="${(mid[1] + ty * 28).toFixed(1)}" text-anchor="middle" fill="#d97706" font-size="11" font-weight="600">dist = ${(s.d / 100).toFixed(2)}</text>
        `;

        const descriptions = [
          `Step 0 (t = 0.000): The ${cBlue('sphere')} starts above and to the left of the ${cPurple("box's top-left corner")}, so GJK's closest witness point on the box is the corner (${cAmber(`dist = ${(s.d/100).toFixed(2)}`)}) and the ${cGreen('separating normal n')} is tilted diagonally. By the Separating Axis Theorem (SAT), the ${cGreen('dashed green plane')} perpendicular to ${cGreen('n')} through the corner separates the two geoms; projecting ${cBlue('Δx₁')} onto ${cGreen('n')} gives v_bound = ${(s.v_bound/100).toFixed(2)}, allowing the sphere to safely advance up to that tilted plane (Δt = ${s.dt.toFixed(3)}).`,
          `Step 1 (t = ${s.t.toFixed(3)}): Advancing to the tilted plane carries the ${cBlue('sphere')} past the corner so it now sits directly above the ${cPurple('flat top face')} with minimal distance ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)}. Because the closest feature on the box is now the top face, the ${cGreen('normal n')} rotates to vertical and the new ${cGreen('dashed green separating plane')} is horizontal. With Δω = 0 and a flat face, one more step Δt = ${s.dt.toFixed(3)} advances straight to target clearance.`,
          `Step 2 — Converged at TOI (t = ${s.t.toFixed(3)}): The ${cBlue('sphere')} lands at target clearance ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)} just above the ${cPurple('top face')} in 2 iterations. Even with zero angular velocity, multiple iterations are needed whenever tangential motion glides a geom past a vertex or edge onto a new feature with a different ${cGreen('separating normal n')}.`
        ];
        return { svg, desc: descriptions[idx] };
      }
    };
  }

  // Scene 1: Complex example (Two boxes with different linear and angular velocities)
  function buildComplexScene() {
    const hx1 = 42, hy1 = 20, r1 = Math.hypot(hx1, hy1);
    const c1_0 = [105, 165], dx1 = [240, -45], th1_0 = -0.65, dw1 = 1.45;

    const hx2 = 28, hy2 = 46, r2 = Math.hypot(hx2, hy2);
    const c2_0 = [545, 115], dx2 = [-205, 40], th2_0 = 0.55, dw2 = -2.10;

    const d_target = 8.0;
    const steps = [];
    let t = 0;
    for (let k = 0; k <= 4; k++) {
      const c1 = [c1_0[0] + t * dx1[0], c1_0[1] + t * dx1[1]];
      const th1 = th1_0 + t * dw1;
      const c2 = [c2_0[0] + t * dx2[0], c2_0[1] + t * dx2[1]];
      const th2 = th2_0 + t * dw2;
      const v1 = boxVerts(c1[0], c1[1], hx1, hy1, th1);
      const v2 = boxVerts(c2[0], c2[1], hx2, hy2, th2);
      const gjk = polyClosest(v1, v2);
      const v_lin = gjk.n[0] * (dx1[0] - dx2[0]) + gjk.n[1] * (dx1[1] - dx2[1]);
      const v_rot = Math.abs(dw1) * r1 + Math.abs(dw2) * r2;
      const v_bound = v_lin + v_rot;
      const dt = k < 4 ? Math.max(0, (gjk.d - d_target) / v_bound) : 0;
      steps.push({ k, t, c1, th1, c2, th2, v1, v2, ...gjk, v_lin, v_rot, v_bound, dt, t_next: t + dt, r1, r2, dx1, dx2 });
      t += dt;
    }

    return {
      steps,
      renderStep(s, idx) {
        const v1End = [s.c1[0] + s.dx1[0] * 0.22, s.c1[1] + s.dx1[1] * 0.22];
        const v2End = [s.c2[0] + s.dx2[0] * 0.22, s.c2[1] + s.dx2[1] * 0.22];
        const mid = [(s.p1[0] + s.p2[0]) / 2, (s.p1[1] + s.p2[1]) / 2];

        const tx = -s.n[1], ty = s.n[0];
        const planeHalf = 78;
        const pl1 = [mid[0] - tx * planeHalf, mid[1] - ty * planeHalf];
        const pl2 = [mid[0] + tx * planeHalf, mid[1] + ty * planeHalf];

        const nBase = [mid[0] - tx * 48, mid[1] - ty * 48];
        const nEnd = [nBase[0] + s.n[0] * 26, nBase[1] + s.n[1] * 26];

        const svg = `
          <!-- Swept center trajectories t in [0, 1] -->
          <line x1="105" y1="165" x2="345" y2="120" class="ca-path1" stroke-width="0.9" stroke-dasharray="3 3"/>
          <line x1="545" y1="115" x2="340" y2="155" class="ca-path2" stroke-width="0.9" stroke-dasharray="3 3"/>

          <!-- Bounding spheres r_bound,1 and r_bound,2 -->
          <circle cx="${s.c1[0].toFixed(1)}" cy="${s.c1[1].toFixed(1)}" r="${s.r1.toFixed(1)}" class="ca-sphere1" stroke-width="0.85" stroke-dasharray="3 3"/>
          <circle cx="${s.c2[0].toFixed(1)}" cy="${s.c2[1].toFixed(1)}" r="${s.r2.toFixed(1)}" class="ca-sphere2" stroke-width="0.85" stroke-dasharray="3 3"/>

          <!-- Active Geoms -->
          <polygon points="${ptsStr(s.v1)}" class="ca-g1-fill" stroke-width="1.2"/>
          <polygon points="${ptsStr(s.v2)}" class="ca-g2-fill" stroke-width="1.2"/>

          <!-- Centers & velocity arrows -->
          <circle cx="${s.c1[0].toFixed(1)}" cy="${s.c1[1].toFixed(1)}" r="2.2" fill="#2563eb"/>
          <line x1="${s.c1[0].toFixed(1)}" y1="${s.c1[1].toFixed(1)}" x2="${v1End[0].toFixed(1)}" y2="${v1End[1].toFixed(1)}" stroke="#3b82f6" stroke-width="1.15" marker-end="url(#ca-arr-b)"/>
          <text x="${(s.c1[0] - 28).toFixed(1)}" y="${(s.c1[1] + s.r1 + 15).toFixed(1)}" class="ca-txt" font-size="11" font-weight="600">geom 1 (t=${s.t.toFixed(2)})</text>

          <!-- Angular velocity arc 1 (clockwise) -->
          <path d="${arcPath(s.c1[0], s.c1[1], s.r1 + 7, -2.3, -1.0, false)}" fill="none" stroke="#3b82f6" stroke-width="1.0" marker-end="url(#ca-arr-b)"/>
          <text x="${(s.c1[0] - 24).toFixed(1)}" y="${(s.c1[1] - s.r1 - 11).toFixed(1)}" fill="#3b82f6" font-size="10.5" font-style="italic">Δω₁, r₁</text>

          <circle cx="${s.c2[0].toFixed(1)}" cy="${s.c2[1].toFixed(1)}" r="2.2" fill="#7c3aed"/>
          <line x1="${s.c2[0].toFixed(1)}" y1="${s.c2[1].toFixed(1)}" x2="${v2End[0].toFixed(1)}" y2="${v2End[1].toFixed(1)}" stroke="#8b5cf6" stroke-width="1.15" marker-end="url(#ca-arr-p)"/>
          <text x="${(s.c2[0] - 28).toFixed(1)}" y="${(s.c2[1] + s.r2 + 15).toFixed(1)}" class="ca-txt" font-size="11" font-weight="600">geom 2 (t=${s.t.toFixed(2)})</text>

          <!-- Angular velocity arc 2 (counter-clockwise, faster) -->
          <path d="${arcPath(s.c2[0], s.c2[1], s.r2 + 7, -0.7, -2.1, true)}" fill="none" stroke="#8b5cf6" stroke-width="1.0" marker-end="url(#ca-arr-p)"/>
          <text x="${(s.c2[0] - 18).toFixed(1)}" y="${(s.c2[1] - s.r2 - 11).toFixed(1)}" fill="#8b5cf6" font-size="10.5" font-style="italic">Δω₂, r₂</text>

          <!-- Separating plane perpendicular to n (SAT) -->
          <line x1="${pl1[0].toFixed(1)}" y1="${pl1[1].toFixed(1)}" x2="${pl2[0].toFixed(1)}" y2="${pl2[1].toFixed(1)}" class="ca-plane" stroke-width="1.0" stroke-dasharray="4 3"/>

          <!-- Separating unit normal n anchored on the separating plane -->
          <line x1="${nBase[0].toFixed(1)}" y1="${nBase[1].toFixed(1)}" x2="${nEnd[0].toFixed(1)}" y2="${nEnd[1].toFixed(1)}" stroke="#10b981" stroke-width="1.25" marker-end="url(#ca-arr-g)"/>
          <text x="${(nEnd[0] + 4).toFixed(1)}" y="${(nEnd[1] + 3).toFixed(1)}" fill="#10b981" font-size="11" font-weight="700">n</text>

          <!-- GJK closest segment dist(t) -->
          <line x1="${s.p1[0].toFixed(1)}" y1="${s.p1[1].toFixed(1)}" x2="${s.p2[0].toFixed(1)}" y2="${s.p2[1].toFixed(1)}" stroke="#f59e0b" stroke-width="1.1" stroke-dasharray="3 2"/>
          <circle cx="${s.p1[0].toFixed(1)}" cy="${s.p1[1].toFixed(1)}" r="2.5" fill="#2563eb"/>
          <circle cx="${s.p2[0].toFixed(1)}" cy="${s.p2[1].toFixed(1)}" r="2.5" fill="#7c3aed"/>
          <text x="${(mid[0] + tx * 26).toFixed(1)}" y="${(mid[1] + ty * 26).toFixed(1)}" text-anchor="middle" fill="#d97706" font-size="11" font-weight="600">dist = ${(s.d / 100).toFixed(2)}</text>
        `;

        const descriptions = [
          `Step 0 (t = 0.000): GJK finds the closest witness points and ${cGreen('separating normal n')} with initial distance ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)}. By the Separating Axis Theorem (SAT), the ${cGreen('dashed green plane')} perpendicular to ${cGreen('n')} separates ${cBlue('geom 1')} and ${cPurple('geom 2')}. Projecting the relative linear displacement onto ${cGreen('n')} (${(s.v_lin/100).toFixed(2)}) and adding both rotational sweep bounds (${(s.v_rot/100).toFixed(2)}) gives v_bound = ${(s.v_bound/100).toFixed(2)}, so both geoms can safely advance to that separating plane by Δt = (dist − dist_target) / v_bound = ${s.dt.toFixed(3)}.`,
          `Step 1 (t = ${s.t.toFixed(3)}): Advancing both geoms linearly and rotationally by Δt = 0.539 brings them up to the previous separating plane without overlapping. Because ${cBlue('geom 1')} rotates clockwise (${cBlue('Δω₁')}) and ${cPurple('geom 2')} rotates faster counter-clockwise (${cPurple('Δω₂')}), their new closest points are still separated by ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)} across a new ${cGreen('dashed green separating plane')} perpendicular to ${cGreen('n')}, allowing a second safe step of Δt = ${s.dt.toFixed(3)}.`,
          `Step 2 (t = ${s.t.toFixed(3)}): At the updated poses, the rotating corners swing closer together and the ${cGreen('SAT separating plane')} tilts as ${cGreen('n')} rotates upward. GJK measures ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)} and bounds the closing rate across the new separating plane by v_bound = ${(s.v_bound/100).toFixed(2)}, advancing both geoms by Δt = ${s.dt.toFixed(3)}.`,
          `Step 3 (t = ${s.t.toFixed(3)}): As the gap narrows to ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)}, the ${cGreen('dashed green separating plane')} aligns with the approaching vertex-face contact and the projected closing rate drops to v_bound = ${(s.v_bound/100).toFixed(2)}, giving a refinement step of Δt = ${s.dt.toFixed(3)}.`,
          `Step 4 — Converged at TOI (t = ${s.t.toFixed(3)}): Minimal distance ${cAmber(`dist = ${(s.d/100).toFixed(2)}`)} reaches the target clearance dist_target just above margin and the search terminates. Keeping this small positive clearance guarantees that a valid ${cGreen('separating plane')} still exists at TOI, so GJK directly returns well-defined witness points and ${cGreen('contact normal n')} without invoking EPA.`
        ];
        return { svg, desc: descriptions[idx] };
      }
    };
  }

  const scenes = [buildBasicScene(), buildComplexScene()];
  let curScene = 0;
  let curStep = 0;

  const gMain = document.getElementById('ca-layer-Main');
  const statusEl = document.getElementById('ca-status-text');
  const stepBtnContainer = document.getElementById('ca-step-btn-container');

  function rebuildStepButtons() {
    const steps = scenes[curScene].steps;
    stepBtnContainer.innerHTML = steps.map((s, i) => {
      const label = i === 0 ? 'Step 0 (t=0)' : (i === steps.length - 1 ? `Step ${i} (TOI)` : `Step ${i}`);
      return `<button type="button" class="ca-btn ca-step-btn" data-step="${i}">${label}</button>`;
    }).join('');
    stepBtnContainer.querySelectorAll('.ca-step-btn').forEach((btn, i) => {
      btn.addEventListener('click', () => render(i));
    });
  }

  function render(idx) {
    const scene = scenes[curScene];
    curStep = idx;
    const s = scene.steps[idx];

    document.querySelectorAll('#mj-ca-widget .ca-scene-btn').forEach((b, i) => {
      b.classList.toggle('ca-active', i === curScene);
    });
    stepBtnContainer.querySelectorAll('.ca-step-btn').forEach((b, i) => {
      b.classList.toggle('ca-active', i === idx);
    });

    const out = scene.renderStep(s, idx);
    gMain.innerHTML = out.svg;
    statusEl.innerHTML = out.desc;
  }

  document.querySelectorAll('#mj-ca-widget .ca-scene-btn').forEach((btn, i) => {
    btn.addEventListener('click', () => {
      curScene = i;
      rebuildStepButtons();
      render(0);
    });
  });

  rebuildStepButtons();
  render(0);
})();
