# 🌡️ Heat Health Watch — Heat-Health Risk Monitoring System

**An OOSE Mini Project | Python · Streamlit · Machine Learning**

Heat Health Watch is an interactive environmental monitoring and risk-assessment application designed around historical temperature, humidity, and air-quality data for Delhi. It combines data visualization, rule-based heat-health risk scoring, Random Forest classification, experimental forecasting, and simulated scenarios in a Streamlit dashboard.

## ✨ Key Features

- **Overview Dashboard:** Displays historical environmental indicators, risk scores, and category distributions.
- **Risk Calculator:** Calculates a project-defined risk score using maximum temperature, relative humidity, and AQI.
- **Historical Trend Analysis:** Visualizes temperature, humidity, pollutant concentrations, and seasonal patterns.
- **Random Forest Classifier:** Classifies historical environmental records into the dataset's Moderate and High labels.
- **Experimental Forecasting:** Generates a synthetic continuation of historical data before estimating future environmental conditions and risk scores.
- **Hourly Risk Simulation:** Displays an illustrative 24-hour risk profile generated from daily data.
- **Synthetic Scenarios:** Generates demonstration records spanning five project-defined risk levels.
- **Data Explorer:** Supports filtering, searching, inspecting, and exporting historical records.
- **CSV Export:** Allows users to download filtered records, forecast estimates, and generated scenarios.
- **Automated Tests:** Includes a pytest test suite for core functionality.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Streamlit | Interactive dashboard |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Random Forest classification |
| Plotly | Interactive visualizations |
| Joblib | Saved model artifacts |
| Pytest | Automated testing |

## 📂 Project Structure

```text
heat_health_watch_project_SRS_Aligned/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── delhi_heatwave_airquality.csv
├── models/
│   ├── rf_risk_classifier.joblib
│   └── rf_metrics.json
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── risk_scoring.py
│   ├── risk_classifier.py
│   ├── forecasting.py
│   ├── hourly_analysis.py
│   ├── synthetic_data.py
│   ├── visualization.py
│   └── advisories.py
├── tests/
├── docs/
└── scripts/
```

### Main Components

- `app.py` — Main Streamlit application.
- `src/config.py` — Configuration and risk-related settings.
- `src/data_loader.py` — Historical data loading and validation.
- `src/preprocessing.py` — Data preparation and feature processing.
- `src/risk_scoring.py` — Rule-based risk-score calculation and categorization.
- `src/risk_classifier.py` — Random Forest classification and model evaluation.
- `src/forecasting.py` — Experimental forecasting functionality.
- `src/hourly_analysis.py` — Simulated hourly risk analysis.
- `src/synthetic_data.py` — Synthetic scenario generation.
- `src/visualization.py` — Charts and visualizations.
- `src/advisories.py` — Category-based advisories.
- `data/` — Historical environmental dataset.
- `models/` — Saved model and evaluation artifacts.
- `tests/` — Automated tests.
- `docs/` — Project documentation.
- `scripts/` — Supporting scripts.

## 🚀 Installation and Setup

### Prerequisites

- Python installed on your system
- pip package manager
- Git (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/zazaaaara/heat-health-watch.git
```

### 2. Navigate to the Project Directory

```bash
cd heat-health-watch/heat_health_watch_project_SRS_Aligned
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Then activate the environment again.

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6. Launch the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal if the dashboard does not open automatically.

### 7. Run the Automated Tests

```bash
python -m pytest -q
```

### Optional: Retrain the Random Forest Model

```bash
python -m src.risk_classifier
```

Run this command only if the module supports direct execution and the required data and dependencies are available.

## 📊 Risk Scoring

The application calculates a project-defined Heat-Health Risk Score using weighted temperature, humidity, and air-quality components.

The implemented formula is represented as:

\[
S = 45N_T + 25N_H + 30\left(\frac{AQI}{500}\right)
\]

Where:

- \(S\) represents the risk score.
- \(N_T\) represents the normalized maximum-temperature component.
- \(N_H\) represents the normalized relative-humidity component.
- \(AQI\) represents the Air Quality Index.

Five demonstration categories are defined in the project configuration: Very Low, Low, Moderate, High, and Severe.

**Note:** Verify that the formula, normalization, and category names match `src/risk_scoring.py` and `src/config.py` before treating them as the final implementation details. These categories and thresholds are project-defined, not official medical or government risk classifications.

## 🤖 Machine Learning

The project uses a Random Forest classifier trained on environmental features from the historical dataset.

The supplied dataset contains two target labels:

- Moderate
- High

The classifier predicts these dataset labels. The separate rule-based scoring module generates the five project-defined risk categories.

The model uses a chronological train-test split to evaluate performance on a later period of historical data. Its metrics measure agreement with the dataset labels; they do not establish clinical accuracy or predict whether an individual will become ill.

## 🔮 Forecasting and Synthetic Data

The forecasting module supports experimental estimates using historical observations and a generated synthetic continuation.

- Historical observations end on **30 November 2024**.
- Synthetic continuation rows are generated estimates, not recorded weather observations.
- Future-date outputs are exploratory scenarios, not official weather forecasts.
- Forecasts beyond approximately 30 days from the continuation are labelled illustrative scenarios.
- Synthetic demonstration scenarios are separate from the historical dataset and are not used to train the Random Forest.
- Hourly profiles are simulations because the source dataset contains daily rather than hourly observations.

The application does not modify the original CSV file during its normal analysis workflow.

## 🧪 Testing

The project includes tests for core application components, including:

- Data loading and validation
- Risk-score calculations
- Random Forest classification
- Forecasting
- Hourly simulation
- Synthetic scenario generation
- Application behaviour

Run the test suite using:

```bash
python -m pytest -q
```

Test results depend on the current code, environment, dependencies, and test execution.

## 🔭 Future Scope

Potential improvements include:

- Integration with live weather and air-quality APIs
- More advanced time-series forecasting
- Additional historical observations and external validation
- Geographic risk visualization
- Automated alerts and notifications
- Database integration for persistent storage
- Broader model testing and rolling-origin evaluation
- Cloud deployment and improved scalability

## ⚠️ Disclaimer

This project is developed for educational and academic purposes. Its risk scores, advisories, synthetic data, and forecasts are not medical diagnoses or official environmental warnings.

For real-world decisions, consult official guidance from the India Meteorological Department (IMD), the Central Pollution Control Board (CPCB), and relevant local health authorities.

