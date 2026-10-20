# Heat-Health Risk Monitoring System

## Overview

The **Heat-Health Risk Monitoring System** is a Python-based application designed to analyze environmental conditions and assess potential heat-health risks. It provides a dashboard for exploring historical data, calculating risk scores, classifying risk categories, and viewing experimental forecasts and simulated scenarios.

## Objectives

- Analyze historical weather and air-quality data.
- Calculate a heat-health risk score using implemented rules.
- Classify risk using a Machine Learning model.
- Visualize environmental trends and risk patterns.
- Explore experimental forecasting and simulated hourly scenarios.
- Provide general advisories based on assessed risk categories.

## Features

- **Overview Dashboard:** Displays a summary of the available environmental data.
- **Risk Calculator:** Calculates risk scores from environmental inputs.
- **Risk Classification:** Uses a saved Random Forest model to predict risk categories.
- **Trend Analysis:** Visualizes historical environmental patterns.
- **Forecasting:** Provides experimental future estimates.
- **Hourly Analysis:** Simulates hourly environmental conditions and risk periods.
- **Synthetic Scenarios:** Generates artificial scenarios for analysis.
- **Data Explorer:** Allows users to inspect the historical dataset.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

## Project Structure

```text
heat_health_watch_project_SRS_Aligned/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── docs/
├── models/
├── scripts/
├── src/
└── tests/
```

### Main Components

- `app.py` — Main Streamlit application.
- `src/` — Data processing, risk scoring, classification, forecasting, visualization, and related modules.
- `data/` — Historical environmental dataset.
- `models/` — Saved Machine Learning model and evaluation artifacts.
- `tests/` — Automated tests for project components.
- `docs/` — Project documentation.
- `scripts/` — Supporting scripts.
- `requirements.txt` — Python dependencies.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/zazaaaara/heat-health-watch.git
```

### 2. Open the Project Directory

```bash
cd heat-health-watch/heat_health_watch_project_SRS_Aligned
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The application should open in a browser using the local Streamlit address displayed in the terminal.

## Testing

The project includes a `tests/` directory containing automated tests for key components, including data handling, risk scoring, classification, forecasting, and application functionality.

Tests can be executed using:

```bash
python -m pytest
```

## Limitations

- The application uses the environmental data available in the project.
- Forecasts and hourly scenarios are experimental estimates or simulations.
- Risk scores and categories are defined by the project's implemented logic.
- Live sensor integration and automatic SMS/email alert delivery are not implemented.
- The results are intended for educational and analytical purposes, not medical diagnosis.

## Future Scope

- Integrate live weather and air-quality APIs.
- Improve forecasting using additional time-series models.
- Expand and validate the risk-classification dataset.
- Add geographical heat-risk visualization.
- Introduce automated alert notifications.
- Deploy the application to a cloud platform.

## Purpose

This project demonstrates the use of Python, data analysis, Machine Learning, visualization, and software engineering practices to build a heat-health risk monitoring application.
