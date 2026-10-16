export const THRESHOLDS = [0.5, 0.9, 0.99];
export const WIDTH = 7;
const EPS = 1e-12;

export function tails(distributions) {
  const result = new Float64Array(distributions.length);
  for (let row = 0; row < distributions.length; row += WIDTH) {
    let sum = 0;
    for (let k = WIDTH - 1; k >= 0; k--) result[row + k] = (sum += distributions[row + k]);
  }
  return result;
}

export function success(first, secondTails, current, rerun, copies) {
  if (copies === 0) return 1;
  let probability = 0;
  for (let k = 0; k < WIDTH; k++) {
    probability += first[current * WIDTH + k] * secondTails[rerun * WIDTH + Math.max(0, copies - k)];
  }
  return Math.min(1, Math.max(0, probability));
}

export function bestSplit(first, secondTails, total, copies) {
  let best = { total, current: total, rerun: 0, copies, probability: -1 };
  if (!secondTails) {
    best.probability = 0;
    for (let k = copies; k < WIDTH; k++) best.probability += first[total * WIDTH + k];
    return best;
  }
  const firstMax = first.length / WIDTH - 1;
  const secondMax = secondTails.length / WIDTH - 1;
  for (let current = Math.min(total, firstMax); current >= Math.max(0, total - secondMax); current--) {
    const probability = success(first, secondTails, current, total - current, copies);
    if (probability > best.probability + EPS) best = { total, current, rerun: total - current, copies, probability };
  }
  return best;
}

export function plan(first, second, { mode, target, budget, maxPulls = 1200 }) {
  const secondTails = second ? tails(second) : null;
  if (mode === 'potential') {
    const result = THRESHOLDS.map(threshold => ({ threshold, unavailable: true, copies: target }));
    for (let total = 0; total <= maxPulls; total++) {
      const best = bestSplit(first, secondTails, total, target);
      for (let i = 0; i < result.length; i++) {
        if (result[i].unavailable && best.probability + EPS >= THRESHOLDS[i]) {
          result[i] = { ...best, threshold: THRESHOLDS[i] };
        }
      }
      if (result.every(item => !item.unavailable)) break;
    }
    return result;
  }
  const base = bestSplit(first, secondTails, budget, 1);
  const result = THRESHOLDS.map(threshold => ({ ...base, copies: 0, threshold }));
  for (let copies = 1; copies < WIDTH; copies++) {
    const best = bestSplit(first, secondTails, budget, copies);
    for (let i = 0; i < result.length; i++) {
      if (best.probability + EPS >= THRESHOLDS[i]) result[i] = { ...best, threshold: THRESHOLDS[i] };
    }
  }
  return result;
}

export function fixedPlanCdf(first, second, copies, currentLimit) {
  const firstTails = tails(first);
  if (!second) {
    return Float64Array.from({ length: first.length / WIDTH }, (_, n) => firstTails[n * WIDTH + copies]);
  }
  const secondTails = tails(second);
  return Float64Array.from({ length: currentLimit + second.length / WIDTH }, (_, n) => (
    n <= currentLimit ? firstTails[n * WIDTH + copies] : success(first, secondTails, currentLimit, n - currentLimit, copies)
  ));
}

export function quantile(cdf, probability) {
  const index = cdf.findIndex(value => value + EPS >= probability);
  return index < 0 ? null : index;
}

export function statistics(cdf) {
  const pmf = Float64Array.from(cdf, (value, i) => Math.max(0, value - (cdf[i - 1] ?? 0)));
  const mass = cdf.at(-1);
  let mean = 0, square = 0;
  pmf.forEach((probability, n) => { mean += n * probability; square += n * n * probability; });
  mean /= mass || 1;
  const variance = Math.max(0, square / (mass || 1) - mean * mean);
  const std = Math.sqrt(variance);
  const lower = Math.max(0, Math.ceil(mean - std));
  const upper = Math.min(cdf.length - 1, Math.floor(mean + std));
  const withinStd = pmf.slice(lower, upper + 1).reduce((a, b) => a + b, 0) / (mass || 1);
  return { pmf, mean, variance, std, lower, upper, withinStd, mass };
}

export function freePulls(banner, ordinary) {
  return (banner === 'rerun' ? [30, 60, 90] : [30]).filter(n => ordinary >= n).length * 10;
}
