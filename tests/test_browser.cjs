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
