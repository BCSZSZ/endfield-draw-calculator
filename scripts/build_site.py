"""Build the GitHub Pages artifact. Runtime needs only a static HTTP server."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from draw_model import MODEL_VERSION, character_distributions
from joint_model import build_joint_history
from weapon_draw import calculate_weapon_full_potential


def build() -> None:
    started = time.monotonic()
    data = ROOT / 'site' / 'data'
    data.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}

    def save(name: str, values: np.ndarray) -> None:
        path = data / name
        path.write_bytes(np.asarray(values, dtype='<f8').tobytes())
        files[name] = {'shape': list(values.shape), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    for banner in ('chartered', 'rerun'):
        for pity in range(80):
            distributions = character_distributions(1200, initial_pity=pity, banner=banner)
            np.testing.assert_allclose(distributions.sum(axis=1), 1.0, atol=3e-13)
            save(f'{banner}-{pity}.bin', distributions)
        print(f'{banner}: all 80 initial pity values exported', flush=True)
    weapons = np.ones((101, 7))
    weapons[0, 1:] = 0
    for target in range(1, 7):
        weapons[:, target] = calculate_weapon_full_potential(target, 100)['finish_probs'].cumsum()
    save('weapons.bin', weapons)
    joint = build_joint_history(progress=lambda draw, mass: print(
        f'joint {draw}/720, retained mass {mass:.12f}', flush=True,
    ))
    if float(joint.metrics[:, 4].min()) < 1 - 1e-9:
        raise ValueError('Joint probability coverage is insufficient')
    save('joint-probabilities.bin', joint.probabilities)
    save('joint-metrics.bin', joint.metrics)
    manifest = {
        'modelVersion': MODEL_VERSION, 'maxPulls': 1200, 'maxPity': 79,
        'maxJointPulls': 720, 'maxWeaponTenPulls': 100, 'copiesCap': 6,
        'jointMinMass': float(joint.metrics[:, 4].min()),
        'dtype': 'little-endian float64', 'files': files,
        'modelSources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ('draw_model.py', 'joint_model.py', 'weapon_draw.py')},
    }
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    vendor = ROOT / 'site' / 'vendor'
    vendor.mkdir(exist_ok=True)
    for filename, url in {
        'plotly-basic-3.3.1.min.js': 'https://cdn.plot.ly/plotly-basic-3.3.1.min.js',
        'plotly-LICENSE.txt': 'https://raw.githubusercontent.com/plotly/plotly.js/v3.3.1/LICENSE',
    }.items():
        path = vendor / filename
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
    print(f'Built site/ in {time.monotonic() - started:.1f}s; {sum(p.stat().st_size for p in data.iterdir()) / 1e6:.2f} MB data', flush=True)


if __name__ == '__main__':
    build()
