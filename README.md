# Hospital Patient Analysis

This project is a hospital patient-arrival analysis dashboard and data pipeline built for educational/statistical exploration. It combines:

- a synthetic hospital arrivals dataset
- a Python data-analysis dashboard using Streamlit
- a FastAPI backend that stores live arrival records in SQLite
- a React + Vite frontend for a lightweight dashboard and manual entry

The code is meant to demonstrate descriptive statistics, probability, hypothesis testing, regression, and forecasting on simulated hospital data. The dataset is not real hospital data.

## Project structure

- `app.py` – Streamlit dashboard for statistical analysis and visualizations
- `backend/main.py` – FastAPI API for receiving and querying arrival counts
- `backend/synthetic_arrival_loader.py` – script to inject synthetic rows automatically
- `frontend/` – React + Vite app that communicates with the backend
- `synthetic_hospital_arrivals.csv` – static 365-day synthetic dataset
- `hospital_analysis.py`, `compare_forecasts.py`, `forecast_analysis.py`, `regression_analysis.py` – supporting analysis scripts
- `Project_Results.txt` – summary of results generated from the dataset

## Architecture overview

### 1) Static analysis layer
The project starts with a CSV file named `synthetic_hospital_arrivals.csv`. It contains daily records like:

- `date`
- `arrivals`
- other derived fields in analysis scripts

This file is used by the Streamlit dashboard and by analysis scripts to calculate statistics and produce charts.

### 2) Live data layer
The FastAPI backend in `backend/main.py` stores incoming arrival observations in SQLite. It exposes endpoints such as:

- `GET /health` – health check
- `POST /api/v1/arrivals` – insert or update a record
- `GET /api/v1/arrivals` – list recent records
- `GET /api/v1/summary` – aggregate summary statistics

The database file is stored at:

- `backend/incoming_counts.db`

The SQLite table is named `arrivals` and has unique constraints for
`(period_start, period, department)` to prevent duplicates for the same time block.

### 3) Frontend UI
The React app in `frontend/` is a dashboard that:

- checks backend health
- loads data from the backend
- allows manual record submission
- shows summary cards and a recent table
- also reads the static CSV file for historical daily chart comparisons

The frontend is not directly connected to SQLite. It talks to the Python FastAPI service over HTTP.

## Are backend and frontend connected?

Yes, but only through the API layer.

The React app uses a dev-server proxy in `frontend/vite.config.js`:

```js
server: {
  proxy: {
    '/api': 'http://127.0.0.1:8000',
    '/health': 'http://127.0.0.1:8000',
  },
}
```

This means when the frontend calls `/api/v1/arrivals`, the Vite proxy forwards it to the running FastAPI backend on port 8000.

The Streamlit app (`app.py`) is separate and is not connected to the React frontend. It reads the CSV file directly and renders statistical pages in its own interface.

## How the app works

### Data input flow
Manual input from the frontend works like this:

1. The user fills in the department and patient count in the React form.
2. The browser sends a POST request to `/api/v1/arrivals`.
3. The FastAPI backend validates the JSON payload.
4. The backend saves the row into SQLite.
5. The frontend refreshes its data and displays the latest record in the table.

Example payload:

```json
{
  "period_start": "2026-10-24T12:00:00+05:30",
  "period": "hour",
  "department": "Emergency",
  "patient_count": 24
}
```

The backend validates:

- `period_start` must be a timezone-aware datetime
- `period` must be either `hour` or `day`
- `department` cannot be empty
- `patient_count` must be >= 0

### Static dataset flow
The CSV file is used for historical and educational analysis. It contains a synthetic 365-day time series of patient arrivals. The `app.py` Streamlit dashboard filters by date, calculates stats, and renders charts. It does not write back to the CSV.

## Running the project

### 1) Install Python dependencies
From the project root:

```bash
pip install -r requirements.txt
```

### 2) Start the backend
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then confirm it is alive:

```bash
curl http://127.0.0.1:8000/health
```

### 3) Start the frontend
Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will usually run on:

- `http://localhost:5173`

### 4) Run the Streamlit analysis app
From the project root:

```bash
streamlit run app.py
```

## How to automate synthetic input

For testing dashboards, simulations, or continuous data flows, you can send synthetic arrival rows automatically.

Use the helper script:

```bash
python backend/synthetic_arrival_loader.py --interval 30 --max-rows 100
```

Parameters:

- `--interval` – wait time between inserted rows in seconds
- `--max-rows` – stop after this many rows; omit to keep running forever
- `--base-url` – default is `http://127.0.0.1:8000`
- `--department` – optional fixed department; otherwise a random department is chosen

Example:

```bash
python backend/synthetic_arrival_loader.py --interval 10 --max-rows 24 --department Emergency
```

This will add one synthetic row every 10 seconds until 24 rows have been submitted.

### What the script does
It generates a JSON payload like:

```json
{
  "period_start": "2026-10-24T13:25:00+05:30",
  "period": "hour",
  "department": "Emergency",
  "patient_count": 18
}
```

Then it sends it to the FastAPI endpoint:

```text
POST /api/v1/arrivals
```

This makes the frontend look like it is receiving live hospital-arrival data in real time.

## Scheduling a recurring synthetic stream

If you want a row every few minutes or hours, you can also run the script using a scheduler:

### Linux/macOS (cron)
```bash
*/5 * * * * cd /path/to/Hospital_Patient_Analysis && python backend/synthetic_arrival_loader.py --interval 300 --max-rows 1
```

### Windows Task Scheduler
Run the same Python command on a repeating schedule.

## Important notes

- This project uses simulated hospital data for educational/analytical purposes.
- The frontend uses live API data and a static historical CSV for comparison.
- The backend database keeps recent records, while the CSV remains the historical baseline.
- The app is designed as a demonstration, not a real production clinical system.

## Summary

This project demonstrates how a small analytics system can be built with:

- Python for data analysis and statistical modeling
- FastAPI for real-time APIs and data storage
- SQLite for lightweight persistence
- React for a dashboard UI
- Streamlit for statistical exploration

It shows both historical analysis and an operational live-input pattern where new data can be inserted automatically at time intervals.
