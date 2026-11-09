// Boids simulation core. No DOM: runs in the browser and in node (sim.test.mjs).
(() => {
  const CAP = 20000, MAXN = 4096, TAU = Math.PI * 2;

  // Struct-of-arrays boid storage.
  const X = new Float32Array(CAP), Y = new Float32Array(CAP);
  const VX = new Float32Array(CAP), VY = new Float32Array(CAP);
  const AX = new Float32Array(CAP), AY = new Float32Array(CAP);
  const SP = new Uint8Array(CAP), DEN = new Uint16Array(CAP);

  // Uniform grid (counting sort, rebuilt every step).
  const cellOf = new Int32Array(CAP), sorted = new Int32Array(CAP), cells = new Int32Array(9);
  let cellStart = new Int32Array(2), cursor = new Int32Array(2);

  // Neighbour scratch buffers filled by gather().
  const nbI = new Int32Array(MAXN), nbDx = new Float32Array(MAXN), nbDy = new Float32Array(MAXN), nbD2 = new Float32Array(MAXN);

  const DEFAULTS = {
    count: 1500, species: 3, predators: 2,
    view: 45, crowd: 16, maxNeighbours: 48, cohesion: 0.006, alignment: 0.06, separation: 0.2,
    maxSpeed: 3.5, minSpeed: 1.5, fear: 110, flee: 0.7, hunt: 180,
    size: 1, wallSize: 24, trails: 0.6, timeScale: 1,
    wrap: true, respawn: true, colorMode: 0,
  };

  const S = {
    CAP, n: 0, W: 800, H: 600, P: { ...DEFAULTS }, DEFAULTS,
    X, Y, VX, VY, SP, DEN,
    preds: [], walls: [], eaten: 0,
    pointer: { x: 0, y: 0, active: false, mode: 'hawk' },
    cols: 1, rows: 1, csx: 1, csy: 1, _species: 0,
  };

  const rand = Math.random;
  const wrapD = (d, L) => (d > L / 2 ? d - L : d < -L / 2 ? d + L : d);
  const dX = (a, b) => (S.P.wrap ? wrapD(a - b, S.W) : a - b);
  const dY = (a, b) => (S.P.wrap ? wrapD(a - b, S.H) : a - b);

  function add(x, y, sp) {
    if (S.n >= CAP) return;
    const i = S.n++, a = rand() * TAU, s = S.P.maxSpeed * 0.7;
    X[i] = x; Y[i] = y; VX[i] = Math.cos(a) * s; VY[i] = Math.sin(a) * s;
    SP[i] = sp ?? (rand() * S.P.species) | 0;
  }

  function remove(i) {
    const l = --S.n;
    X[i] = X[l]; Y[i] = Y[l]; VX[i] = VX[l]; VY[i] = VY[l]; SP[i] = SP[l]; DEN[i] = DEN[l];
  }

  // Reconcile state with params (count, species, predator count).
  function sync() {
    const P = S.P;
    P.count = Math.min(P.count, CAP);
    P.minSpeed = Math.min(P.minSpeed, P.maxSpeed);
    while (S.n < P.count) add(rand() * S.W, rand() * S.H);
    if (S.n > P.count) S.n = P.count;
    if (S._species !== P.species) {
      for (let i = 0; i < S.n; i++) SP[i] = i % P.species;
      S._species = P.species;
    }
    while (S.preds.length < P.predators)
      S.preds.push({ x: rand() * S.W, y: rand() * S.H, vx: 0, vy: 0, cool: 0, wa: rand() * TAU });
    S.preds.length = P.predators;
  }

  function reset() { S.n = 0; S.preds = []; S.eaten = 0; S._species = 0; sync(); }

  function buildGrid() {
    const v = Math.max(S.P.view, 4);
    // Cells at least `view` wide, so a 3x3 block always covers the view radius (also across the wrap seam).
    const cols = Math.max(1, Math.floor(S.W / v)), rows = Math.max(1, Math.floor(S.H / v)), nc = cols * rows;
    S.cols = cols; S.rows = rows; S.csx = S.W / cols; S.csy = S.H / rows;
    if (cellStart.length < nc + 1) { cellStart = new Int32Array(nc + 1); cursor = new Int32Array(nc + 1); }
    cellStart.fill(0, 0, nc + 1);
    for (let i = 0; i < S.n; i++) {
      const cx = Math.min(cols - 1, Math.max(0, Math.floor(X[i] / S.csx)));
      const cy = Math.min(rows - 1, Math.max(0, Math.floor(Y[i] / S.csy)));
      const c = cy * cols + cx;
      cellOf[i] = c; cellStart[c + 1]++;
    }
    for (let c = 0; c < nc; c++) cellStart[c + 1] += cellStart[c];
    cursor.set(cellStart.subarray(0, nc));
    for (let i = 0; i < S.n; i++) sorted[cursor[cellOf[i]]++] = i;
    // Reorder boids into cell order so neighbours sit close in memory (cache locality).
    for (const [A, T] of PERM) {
      for (let k = 0; k < S.n; k++) T[k] = A[sorted[k]];
      A.set(T.subarray(0, S.n));
    }
    for (let k = 0; k < S.n; k++) TC[k] = cellOf[sorted[k]];
    cellOf.set(TC.subarray(0, S.n));
    for (let k = 0; k < S.n; k++) sorted[k] = k;
  }

  const F = new Float32Array(CAP), TC = new Int32Array(CAP);
  const PERM = [[X, F], [Y, F], [VX, F], [VY, F], [SP, new Uint8Array(CAP)], [DEN, new Uint16Array(CAP)]];

  // Fill nb* buffers with boids within `view` of boid i. Returns count. Requires buildGrid().
  function gather(i) {
    const { cols, rows, W, H } = S, wrap = S.P.wrap, v2 = S.P.view * S.P.view, xi = X[i], yi = Y[i];
    const c = cellOf[i], cx = c % cols, cy = (c / cols) | 0, lim = Math.min(MAXN, S.P.maxNeighbours | 0);
    let nc = 0;
    for (let oy = -1; oy <= 1; oy++) {
      let r = cy + oy;
      if (r < 0 || r >= rows) { if (!wrap) continue; r = (r + rows) % rows; }
      for (let ox = -1; ox <= 1; ox++) {
        let q = cx + ox;
        if (q < 0 || q >= cols) { if (!wrap) continue; q = (q + cols) % cols; }
        const id = r * cols + q;
        let dup = false;
        for (let k = 0; k < nc; k++) if (cells[k] === id) { dup = true; break; }
        if (!dup) cells[nc++] = id;
      }
    }
    let m = 0;
    for (let k = 0; k < nc; k++) {
      const id = cells[k];
      for (let s = cellStart[id], e = cellStart[id + 1]; s < e; s++) {
        const j = sorted[s];
        if (j === i) continue;
        let dx = X[j] - xi, dy = Y[j] - yi;
        if (wrap) { dx = wrapD(dx, W); dy = wrapD(dy, H); }
        const d2 = dx * dx + dy * dy;
        if (d2 >= v2) continue;
        if (m === lim) return m; // starlings track ~7 neighbours; capping also bounds cost in dense clumps
        nbI[m] = j; nbDx[m] = dx; nbDy[m] = dy; nbD2[m] = d2; m++;
      }
    }
    return m;
  }

  // Scratch outputs (no per-call allocation in hot loops).
  const W2 = new Float64Array(2), MV = new Float64Array(4);

  function avoidWalls(x, y, margin) {
    let ax = 0, ay = 0;
    for (const o of S.walls) {
      const dx = x - o.x, dy = y - o.y, d = Math.hypot(dx, dy), m = o.r + margin;
      if (d < m && d > 0) { const push = ((m - d) / margin) * 0.9; ax += (dx / d) * push; ay += (dy / d) * push; }
    }
    W2[0] = ax; W2[1] = ay;
  }

  function edgeForce(x, L) {
    const M = 60;
    return x < M ? ((M - x) / M) * 0.5 : x > L - M ? -((x - (L - M)) / M) * 0.5 : 0;
  }

  // Integrate one body into MV = [x, y, vx, vy].
  function move(x, y, vx, vy, dt, max, min) {
    const { W, H } = S;
    let s = Math.hypot(vx, vy);
    if (s === 0) { vx = min || 0.1; s = vx; }
    if (s > max) { vx *= max / s; vy *= max / s; } else if (s < min) { vx *= min / s; vy *= min / s; }
    x += vx * dt; y += vy * dt;
    if (S.P.wrap) {
      x = ((x % W) + W) % W; y = ((y % H) + H) % H;
    } else {
      if (x < 0) { x = 0; vx = Math.abs(vx); } else if (x >= W) { x = W - 0.01; vx = -Math.abs(vx); }
      if (y < 0) { y = 0; vy = Math.abs(vy); } else if (y >= H) { y = H - 0.01; vy = -Math.abs(vy); }
    }
    for (const o of S.walls) {
      const dx = x - o.x, dy = y - o.y, d2 = dx * dx + dy * dy;
      if (d2 < o.r * o.r) { const d = Math.sqrt(d2) || 0.001; x = o.x + (dx / d) * o.r; y = o.y + (dy / d) * o.r; }
    }
    MV[0] = x; MV[1] = y; MV[2] = vx; MV[3] = vy;
  }

  function step(dt = 1) {
    const P = S.P, n = S.n, cr = P.crowd, cr2 = cr * cr, fear2 = P.fear * P.fear, ptr = S.pointer;
    buildGrid();

    const threats = S.preds.slice();
    if (ptr.active && ptr.mode === 'hawk') threats.push(ptr);
    const lure = ptr.active && ptr.mode === 'lure';

    for (let i = 0; i < n; i++) {
      const m = gather(i), sp = SP[i], xi = X[i], yi = Y[i];
      let cx = 0, cy = 0, avx = 0, avy = 0, sx = 0, sy = 0, same = 0;
      for (let k = 0; k < m; k++) {
        const j = nbI[k], dx = nbDx[k], dy = nbDy[k], d2 = nbD2[k];
        if (SP[j] === sp) { same++; cx += dx; cy += dy; avx += VX[j]; avy += VY[j]; }
        if (d2 < cr2 && d2 > 0) { const d = Math.sqrt(d2), w = (cr - d) / (cr * d); sx -= dx * w; sy -= dy * w; }
      }
      DEN[i] = m;
      let ax = sx * P.separation, ay = sy * P.separation;
      if (same) {
        ax += (cx / same) * P.cohesion + (avx / same - VX[i]) * P.alignment;
        ay += (cy / same) * P.cohesion + (avy / same - VY[i]) * P.alignment;
      }
      for (const t of threats) {
        const dx = dX(xi, t.x), dy = dY(yi, t.y), d2 = dx * dx + dy * dy;
        if (d2 < fear2 && d2 > 0) { const d = Math.sqrt(d2), w = (P.flee * (1 - d / P.fear)) / d; ax += dx * w; ay += dy * w; }
      }
      if (lure) {
        const dx = dX(ptr.x, xi), dy = dY(ptr.y, yi), d = Math.hypot(dx, dy);
        if (d > 1 && d < 300) { ax += (dx / d) * 0.12; ay += (dy / d) * 0.12; }
      }
      if (S.walls.length) { avoidWalls(xi, yi, P.view * 0.6 + 6); ax += W2[0]; ay += W2[1]; }
      if (!P.wrap) { ax += edgeForce(xi, S.W); ay += edgeForce(yi, S.H); }
      AX[i] = ax; AY[i] = ay;
    }

    for (let i = 0; i < n; i++) {
      move(X[i], Y[i], VX[i] + AX[i] * dt, VY[i] + AY[i] * dt, dt, P.maxSpeed, P.minSpeed);
      X[i] = MV[0]; Y[i] = MV[1]; VX[i] = MV[2]; VY[i] = MV[3];
    }

    stepPredators(dt);

    if (P.respawn) {
      for (let k = Math.min(P.count - S.n, 3); k > 0; k--) {
        // Enter from a random edge.
        const e = rand() * 4 | 0, u = rand();
        add(e === 0 ? 0 : e === 1 ? S.W - 1 : u * S.W, e === 2 ? 0 : e === 3 ? S.H - 1 : u * S.H);
      }
    }
  }

  function stepPredators(dt) {
    const P = S.P, n = S.n, eaten = new Set(), reach = 6 + 4 * P.size;
    for (const p of S.preds) {
      let best = -1, bd = P.hunt * P.hunt;
      if (p.cool <= 0) {
        // ponytail: O(preds * n) target search, fine for <=10 predators; use the grid if predators scale up
        for (let i = 0; i < n; i++) {
          const dx = dX(X[i], p.x), dy = dY(Y[i], p.y), d2 = dx * dx + dy * dy;
          if (d2 < bd && !eaten.has(i)) { bd = d2; best = i; }
        }
      }
      const pmax = P.maxSpeed * (p.cool > 0 ? 0.55 : 1.15);
      let tx, ty;
      if (best >= 0) {
        const dx = dX(X[best], p.x), dy = dY(Y[best], p.y), d = Math.sqrt(bd) || 1;
        tx = (dx / d) * pmax; ty = (dy / d) * pmax;
        if (d < reach) { eaten.add(best); p.cool = 90; S.eaten++; }
      } else {
        p.wa += (rand() - 0.5) * 0.3 * dt;
        tx = Math.cos(p.wa) * pmax * 0.7; ty = Math.sin(p.wa) * pmax * 0.7;
      }
      let ax = (tx - p.vx) * 0.05, ay = (ty - p.vy) * 0.05;
      for (const q of S.preds) {
        if (q === p) continue;
        const dx = dX(p.x, q.x), dy = dY(p.y, q.y), d = Math.hypot(dx, dy);
        if (d > 0 && d < 70) { ax += (dx / d) * 0.08; ay += (dy / d) * 0.08; }
      }
      if (S.walls.length) { avoidWalls(p.x, p.y, 30); ax += W2[0]; ay += W2[1]; }
      if (!P.wrap) { ax += edgeForce(p.x, S.W); ay += edgeForce(p.y, S.H); }
      move(p.x, p.y, p.vx + ax * dt, p.vy + ay * dt, dt, pmax, 0.3);
      p.x = MV[0]; p.y = MV[1]; p.vx = MV[2]; p.vy = MV[3];
      p.cool -= dt;
    }
    // Remove highest index first so swap-with-last never moves a boid still pending removal.
    [...eaten].sort((a, b) => b - a).forEach(remove);
  }

  function addWall(x, y) {
    const r = S.P.wallSize, last = S.walls[S.walls.length - 1];
    if (last && Math.hypot(last.x - x, last.y - y) < r * 0.6) return;
    S.walls.push({ x, y, r });
  }

  function erase(x, y, r) {
    S.walls = S.walls.filter(o => Math.hypot(o.x - x, o.y - y) > o.r + r);
  }

  function resize(W, H) { S.W = Math.max(1, W); S.H = Math.max(1, H); }

  const neighbours = i => Array.from(nbI.subarray(0, gather(i))).sort((a, b) => a - b);

  globalThis.Sim = { S, add, remove, sync, reset, step, buildGrid, neighbours, addWall, erase, resize };
})();
