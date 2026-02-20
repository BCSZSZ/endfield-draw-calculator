# EndfieldP

Two gacha calculators and one combined Streamlit app:

- Character draw DP: `myturnDRAW6up.py`
- Weapon draw DP: `weapon_draw.py`
- Combined Streamlit app: `streamlit_all_app.py`

## Online Demo

Direct access to the deployed Streamlit app:

- https://endfield-draw-calculator.streamlit.app/

## Setup (Windows)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install numpy pandas pyarrow plotly streamlit
```

For joint character+weapon precompute cache (Parquet):

```powershell
python precompute_joint_results.py
```

This generates/updates `precomputed/joint_results_dataset/` (Parquet dataset) for fast lookup in Streamlit.

## Rule Terminology (Unified)

To keep UI and docs consistent, use the following wording:

- **Draw outcome**: Result produced by the draw process itself (including pity-triggered draw outcomes).
- **Direct gift**: Result granted by milestone/reward rules, not produced by a draw.

### Character rules

- **80-pull small pity** is a draw outcome.
- **120-pull first-cycle big pity** is also a draw outcome.
- **Every 240 pulls** gives a direct gift (UP token/duplicate), not a draw outcome.

### Weapon rules

- **At 18 ten-pulls and every 16 ten-pulls after that** (18, 34, 50, ...) are direct gifts of UP weapon, not draw outcomes.

### Joint model quota rule

- Weapon quota (`+100/+10/+1` from 6★/5★/4★ character results) is accumulated from draw outcomes.
- Direct gifts (for example, the 240-pull character cycle gift) do not add extra weapon quota.

## Run

### Streamlit (recommended)

```powershell
streamlit run streamlit_all_app.py
```

### CLI (optional)

```powershell
python myturnDRAW6up.py
```

```powershell
python weapon_draw.py
```
