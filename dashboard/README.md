# PrognosAI Dashboard

The frontend dashboard for PrognosAI, an industrial predictive-maintenance project based on the NASA C-MAPSS FD001 turbofan engine degradation dataset.

## Technology Stack
- React
- TypeScript
- Vite

## Prerequisites
- Node.js and npm
- Python environment with project dependencies installed

## Installation
Run from this directory:
```powershell
npm install
```

## Running the Application
Start the backend from the project root:
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Start the dashboard from this directory:
```powershell
npm run dev
```

## Production Build
```powershell
npm run build
```

## Backend Integration
The dashboard communicates with the FastAPI backend through the /api proxy configured in Vite.

## Dataset
The NASA C-MAPSS FD001 dataset is maintained locally and is not included in this repository.

## Research Disclaimer
PrognosAI is an academic predictive-maintenance prototype. Model results are based on simulated turbofan engine data and are not validated performance results for real industrial equipment.
