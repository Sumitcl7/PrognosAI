# PrognosAI

## Uncertainty-Aware Predictive Maintenance and Decision Support System

PrognosAI is an academic predictive-maintenance prototype using the NASA C-MAPSS FD001 simulated turbofan engine degradation dataset. It combines Remaining Useful Life (RUL) prediction, conformal uncertainty estimation, drift monitoring, and risk-based maintenance decision support.

## Technology Stack
- Python, NumPy, Pandas, SciPy
- Scikit-learn and Random Forest
- FastAPI and Uvicorn
- React, TypeScript, Vite
- Pytest and Joblib

## Project Structure
- src/preprocessing - Dataset exploration and RUL generation
- src/models - Model training and evaluation
- src/uncertainty - Conformal prediction
- src/drift - Drift monitoring
- src/decision - Maintenance decision logic
- src/api - FastAPI backend
- dashboard - React frontend
- tests - Automated tests
- artifacts - Selected runtime model and evaluation artifacts

## Local Setup

Run commands from the project root, E:\PrognosAI.

Install Python dependencies:
```powershell
pip install -r requirements.txt
```

Install frontend dependencies:
```powershell
cd dashboard
npm install
cd ..
```

## Run the Application

Start the backend from the project root:
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the dashboard in another terminal:
```powershell
cd E:\PrognosAI\dashboard
npm run dev
```

API documentation: http://127.0.0.1:8000/docs

## Tests
Run from the project root:
```powershell
python -m pytest tests -v
```

## Model Evaluation

Random Forest baseline results:

| Metric | Validation | Official FD001 Test |
|---|---:|---:|
| MAE | 12.3359 | 11.8056 |
| RMSE | 17.0349 | 16.9546 |
| R2 | 0.8332 | 0.8210 |

Official test NASA asymmetric score: 836.3887.

RUL predictions are capped at 125 cycles for the capped evaluation.

## Research Limitations
The NASA C-MAPSS dataset contains simulated turbofan engine data. Results do not establish performance on real industrial equipment. Maintenance recommendations are experimental decision-support outputs and require further validation before operational use.

Dataset files and local development environments are not included in version control.
