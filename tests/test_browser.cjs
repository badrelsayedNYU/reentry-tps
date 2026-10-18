// Exercise the actual model embedded in the app, without a DOM or third-party test packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../docs/index.html'), 'utf8');
const source = html.match(/<script id="tps-src">([\s\S]*?)<\/script>/)[1];
const TPS = vm.runInNewContext(`${source}\nTPS;`);
const trajectory = TPS.fly({ V_E: 7800, gamma_E_deg: -1.5, beta: 400, LD: 0.3, Rn: 4 });
const c = { shield: 'li900', Rn: 4 };

for (const earlyStop of [false, true]) {
  test(`an unconfirmed peak is refused, earlyStop=${earlyStop}`, () => {
    assert.throws(() => TPS.simulate(c, 0.07, trajectory, {
      soak: 1, maxSoak: 2, earlyStop,
    }), /Bondline peak not confirmed/);
  });
}

test('sizing cannot turn an unconfirmed peak into a thickness', () => {
  assert.throws(() => TPS.size(c, trajectory, { maxSoak: 1 }), /Bondline peak not confirmed/);
});

test('a larger initial window still respects the maximum soak time', () => {
  assert.throws(() => TPS.simulate(c, 0.07, trajectory, {
    soak: 14400, maxSoak: 1,
  }), /after 1 s of soak-back/);
});

test('zero and unbounded soak windows are refused', () => {
  for (const opts of [{ soak: 0 }, { maxSoak: 0 }, { maxSoak: Infinity }]) {
    assert.throws(() => TPS.simulate(c, 0.07, trajectory, opts), /finite and positive/);
  }
});

for (const [shield, expected] of [['li900', 0.0703], ['li2200', 0.0639], ['frci12', 0.0721]]) {
  test(`baseline ${shield} still sizes to the published result with a captured peak`, () => {
    const materialCase = { ...c, shield };
    const sized = TPS.size(materialCase, trajectory);
    assert.ok(Math.abs(sized.thickness - expected) <= 1e-4);
    const result = TPS.simulate(materialCase, sized.thickness, trajectory);
    assert.equal(result.captured, true);
    assert.ok(result.peakTb <= 430.1);
  });
}

test('saved wall profiles align with time samples and wall interfaces', () => {
  const r = TPS.simulate(c, 0.0703, trajectory, { saveProfiles: true, keepEvery: 50 });
  assert.equal(r.hist.profiles.length, r.hist.t.length);
  assert.equal(r.hist.depth.length, r.hist.profiles[0].length);
  assert.equal(r.hist.depth[0], 0);
  assert.ok(Math.abs(r.hist.depth.at(-1) - 0.0725) < 1e-10);
  assert.ok(r.hist.profiles.every(p => p.every(Number.isFinite)));
});

test('advanced wall settings change the grid and mass rather than just labels', () => {
  const custom = { ...c, shieldCells: 20, bondThickness: 0.001, skinThickness: 0.004 };
  const wall = TPS.caseWall(custom, 0.07);
  assert.equal(wall.N, 30);
  assert.ok(Math.abs(wall.thickness - 0.075) < 1e-12);
  assert.ok(TPS.arealMass(custom, 0.07) > TPS.arealMass(c, 0.07));
});

test('hot-wall correction reduces applied heating on the same fixed shield', () => {
  const cold = TPS.simulate(c, 0.0703, trajectory, { earlyStop: true, soak: 14400 });
  const hot = TPS.simulate({ ...c, hotWall: true }, 0.0703, trajectory, { earlyStop: true, soak: 14400 });
  assert.ok(hot.peakQ < cold.peakQ);
  assert.ok(hot.peakTs < cold.peakTs);
  assert.ok(hot.peakTb < cold.peakTb);
});

test('parameter study computes separate cases and reports failed points', () => {
  const rows = TPS.sweep({ ...c, V_E: 7800, gamma_E_deg: -1.5, beta: 400, LD: 0.3,
    maxSoak: 1 }, 'Rn', [2, 4]);
  assert.equal(rows.length, 2);
  assert.ok(rows.every(row => row.error && row.thickness === undefined));
  assert.throws(() => TPS.sweep(c, 'unknown', [1, 2]), /supported parameter/);
  assert.throws(() => TPS.sweep(c, 'Rn', [0, 2]), /between/);
});


test('a nose-radius study confirms every peak and sizes thinner tiles at larger radius', () => {
  const rows = TPS.sweep({ ...c, V_E: 7800, gamma_E_deg: -1.5, beta: 400, LD: 0.3 }, 'Rn', [2, 4]);
  assert.ok(rows.every(row => !row.error && row.captured && row.peakTb <= 430.1));
  assert.ok(rows[0].thickness > rows[1].thickness);
  assert.ok(rows[0].peakQ > rows[1].peakQ);
});
