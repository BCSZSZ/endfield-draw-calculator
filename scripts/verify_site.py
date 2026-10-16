"""Validate built data and write independent NumPy planner answers for the JS tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'site' / 'data'


def main() -> None:
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['modelSources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Rebuild stale model data: {name}'
    arrays = {}
    for name, meta in manifest['files'].items():
        raw = (DATA / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta['sha256'], name
        values = np.frombuffer(raw, dtype='<f8').reshape(meta['shape'])
        assert np.isfinite(values).all() and values.min() >= -1e-14, name
        arrays[name] = values
        if name.startswith(('chartered-', 'rerun-')):
            np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=3e-13)
            tails = values[:, ::-1].cumsum(axis=1)[:, ::-1]
            assert np.diff(tails, axis=0).min() >= -1e-13, name
    joint = arrays['joint-probabilities.bin']
    first = arrays['chartered-0.bin']
    np.testing.assert_allclose(joint[:, :, 0], first[:721, ::-1].cumsum(axis=1)[:, ::-1], atol=1e-10)
    assert joint.max() <= 1 + 1e-12
    assert np.diff(joint, axis=0).min() >= -1e-10
    assert (joint[:, 1:, 1:] <= joint[:, 1:, :1] + 1e-12).all()
    assert (joint[:, 1:, 1:] <= joint[:, :1, 1:] + 1e-12).all()
    assert arrays['joint-metrics.bin'][:, 4].min() >= 1 - 1e-9
    fixtures = []
    for pity, rerun_pity, include, mode, value in [
        (0, 0, False, 'potential', 1), (0, 0, True, 'potential', 6),
        (79, 39, True, 'potential', 3), (0, 0, False, 'potential', 6),
        (0, 0, True, 'budget', 0), (0, 0, True, 'budget', 120),
        (0, 0, True, 'budget', 300), (79, 39, True, 'budget', 480),
        (0, 0, False, 'budget', 300), (0, 0, True, 'budget', 1200),
    ]:
        first = arrays[f'chartered-{pity}.bin']
        second = arrays[f'rerun-{rerun_pity}.bin']
        second_tails = second[:, ::-1].cumsum(axis=1)[:, ::-1]

        def best(total: int, copies: int) -> dict:
            a = np.arange(total, -1, -1) if include else np.array([total])
            b = total - a
            if include:
                probabilities = sum(
                    first[a, k] * second_tails[b, max(0, copies - k)] for k in range(7)
                )
            else:
                probabilities = first[a, copies:].sum(axis=1)
            index = int(np.flatnonzero(probabilities >= probabilities.max() - 1e-12)[0])
            return dict(total=total, current=int(a[index]), rerun=int(b[index]),
                        copies=copies, probability=float(probabilities[index]))

        answers = []
        for q in (.5, .9, .99):
            if mode == 'potential':
                for total in range(1201):
                    answer = best(total, value)
                    if answer['probability'] + 1e-12 >= q:
                        break
            else:
                answer = {**best(value, 1), 'copies': 0}
                for copies in range(1, 7):
                    candidate = best(value, copies)
                    if candidate['probability'] + 1e-12 >= q:
                        answer = candidate
            answers.append({**answer, 'threshold': q})
        fixtures.append(dict(pity=pity, rerunPity=rerun_pity, includeRerun=include,
                             config=dict(mode=mode, target=value if mode == 'potential' else 6,
                                         budget=value if mode == 'budget' else 300), answers=answers))
    output = ROOT / 'tests' / '.generated'
    output.mkdir(exist_ok=True)
    (output / 'planner-fixtures.json').write_text(json.dumps(fixtures, indent=2), encoding='utf-8')
    print(f'Validated {len(arrays)} data files and all 25,920 joint scenarios; wrote {len(fixtures)} independent planner cases.')


if __name__ == '__main__':
    main()
