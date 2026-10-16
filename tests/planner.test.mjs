import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { plan, quantile, fixedPlanCdf, statistics, freePulls } from '../site/planner.mjs';

const fixtures = JSON.parse(readFileSync(new URL('./.generated/planner-fixtures.json', import.meta.url)));
function load(name) {
  const buffer = readFileSync(new URL(`../site/data/${name}`, import.meta.url));
  return Float64Array.from({ length: buffer.length / 8 }, (_, i) => buffer.readDoubleLE(i * 8));
}

for (const fixture of fixtures) {
  test(`Python oracle: ${JSON.stringify({ ...fixture.config, pity: fixture.pity, rerun: fixture.includeRerun })}`, () => {
    const first = load(`chartered-${fixture.pity}.bin`);
    const second = fixture.includeRerun ? load(`rerun-${fixture.rerunPity}.bin`) : null;
    const actual = plan(first, second, fixture.config);
    actual.forEach((item, i) => {
      const expected = fixture.answers[i];
      for (const key of ['total', 'current', 'rerun', 'copies', 'threshold']) assert.equal(item[key], expected[key], key);
      assert.ok(Math.abs(item.probability - expected.probability) < 1e-11);
      assert.equal(item.current + item.rerun, item.total);
      if (item.copies) {
        const cdf = fixedPlanCdf(first, second, item.copies, item.current);
        assert.ok(Math.abs(cdf[item.total] - item.probability) < 1e-11);
        assert.ok(cdf.at(-1) > 1 - 1e-12);
        assert.ok(Math.abs(statistics(cdf).pmf.reduce((a, b) => a + b, 0) - 1) < 1e-12);
      }
    });
    if (fixture.config.mode === 'potential') assert.ok(actual[0].total <= actual[1].total && actual[1].total <= actual[2].total);
    else assert.ok(actual[0].copies >= actual[1].copies && actual[1].copies >= actual[2].copies);
  });
}

test('discrete quantiles do not interpolate or invent an out-of-range answer', () => {
  assert.equal(quantile([0, .2, .2, .9], .5), 3);
  assert.equal(quantile([0, .2, .2, .9], .99), null);
  assert.equal(quantile([0, .2], 0), 0);
});

test('reward counts include only reached milestones', () => {
  assert.equal(freePulls('chartered', 90), 10);
  assert.equal(freePulls('rerun', 59), 10);
  assert.equal(freePulls('rerun', 60), 20);
  assert.equal(freePulls('rerun', 90), 30);
  assert.equal(freePulls('rerun', 1200), 30);
});
