# EndfieldP

Two gacha calculators and one combined Streamlit app:

- Character draw DP: `myturnDRAW6up.py`
- Weapon draw DP: `weapon_draw.py`
- Combined Streamlit app: `streamlit_all_app.py`

## Setup (Windows)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install numpy pandas plotly streamlit
```

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
