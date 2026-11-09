// Run: node sim.test.mjs
import assert from 'node:assert/strict';
import './sim.js';

const { S, sync, reset, step, buildGrid, neighbours, addWall } = globalThis.Sim;

function brute(i) {
  const out = [], v2 = S.P.view ** 2;
  for (let j = 0; j < S.n; j++) {
    if (j === i) continue;
    let dx = S.X[j] - S.X[i], dy = S.Y[j] - S.Y[i];
    if (S.P.wrap) {
      if (dx > S.W / 2) dx -= S.W; else if (dx < -S.W / 2) dx += S.W;
      if (dy > S.H / 2) dy -= S.H; else if (dy < -S.H / 2) dy += S.H;
    }
    if (dx * dx + dy * dy < v2) out.push(j);
  }
  return out;
}

// 1. Grid neighbour search == brute force, incl. tiny worlds (cols < 3) and wrap seams.
for (const [W, H, view, wrap] of [[800, 600, 50, true], [800, 600, 50, false], [90, 70, 40, true], [1000, 130, 150, true], [37, 500, 45, false]]) {
  Object.assign(S.P, { ...S.DEFAULTS, view, wrap, count: 600, predators: 0, maxNeighbours: 4096 });
  S.W = W; S.H = H; reset();
  buildGrid();
  for (let i = 0; i < S.n; i++) assert.deepEqual(neighbours(i), brute(i), `grid mismatch W=${W} H=${H} view=${view} wrap=${wrap} i=${i}`);
}

// 2. Long run with walls, predators, both edge modes: everything stays finite and in bounds.
for (const wrap of [true, false]) {
  Object.assign(S.P, { ...S.DEFAULTS, wrap, count: 2000, predators: 4, respawn: false });
  S.W = 900; S.H = 700; S.walls = []; reset();
  for (let k = 0; k < 20; k++) addWall(200 + k * 25, 350);
  for (let t = 0; t < 600; t++) step(t % 7 === 0 ? 2.5 : 1);
  for (let i = 0; i < S.n; i++) {
    assert.ok(Number.isFinite(S.X[i] + S.Y[i] + S.VX[i] + S.VY[i]), `NaN at ${i}`);
    assert.ok(S.X[i] >= 0 && S.X[i] < S.W && S.Y[i] >= 0 && S.Y[i] < S.H, `out of bounds at ${i}`);
  }
  // 3. Predators eat, and eaten boids are actually removed.
  assert.ok(S.eaten > 0, 'predators never ate');
  assert.equal(S.n, 2000 - S.eaten);
}

// 4. Respawn refills, sync trims.
S.P.respawn = true;
for (let t = 0; t < 400; t++) step();
assert.ok(S.n >= S.P.count - 3);
S.P.count = 100; sync();
assert.equal(S.n, 100);

console.log('ok');
