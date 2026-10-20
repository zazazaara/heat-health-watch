# 🌡️ Heat Health Watch — Heat-Health Risk Monitoring System

**An OOSE Mini Project | Python · Streamlit · Machine Learning**

Heat Health Watch is an interactive environmental monitoring and risk-assessment application designed around historical temperature, humidity, and air-quality data for Delhi. It combines data visualization, rule-based heat-health risk scoring, Random Forest classification, experimental forecasting, and simulated scenarios in a Streamlit dashboard.

## ✨ Key Features

- **Overview Dashboard:** Monitor historical environmental indicators, risk scores, and category distributions.
- **Risk Calculator:** Calculate a project-defined risk score using maximum temperature, relative humidity, and AQI.
- **Historical Trend Analysis:** Visualize temperature, humidity, pollutant concentrations, and seasonal patterns.
- **Random Forest Classifier:** Classify historical environmental records into Moderate and High dataset labels.
- **Experimental Forecasting:** Generate a synthetic continuation of historical data before estimating future environmental conditions and risk scores.
- **Hourly Risk Simulation:** Explore an illustrative 24-hour risk profile generated from daily data.
- **Synthetic Scenarios:** Generate demonstration records spanning five project-defined risk levels.
- **Data Explorer:** Filter, search, inspect, and export historical records.
- **CSV Export:** Download filtered records, forecast estimates, and generated scenarios.
- **Automated Tests:** Includes a pytest test suite for core functionality.

## 🛠️ Tech Stack

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

## 🚀 Installation and Setup

### Prerequisites

- Python installed on your system
- pip package manager
- Git (if cloning the repository)

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd heat_health_watch_project_SRS_Aligned
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal if the dashboard does not open automatically.

### 6. Run the automated tests

```bash
python -m pytest -q
```

### Optional: Retrain the Random Forest model

```bash
python -m src.risk_classifier
```

## 📊 Risk Scoring

The application calculates a project-defined Heat-Health Risk Score using the following formula:

\[
S = 45N_T + 25N_H + 30\left(\frac{AQI}{500}\right)
\]

Where:

- \(S\) represents the risk score.
- \(N_T\) is the normalized maximum-temperature component.
- \(N_H\) is the normalized relative-humidity component.
- \(AQI\) is the Air Quality Index.

The score uses weighted temperature, humidity, and air-quality components. Five demonstration categories are defined in the project configuration: Very Low, Low, Moderate, High, and Severe.

These categories and thresholds are project-defined, not official medical or government risk classifications.

## 🤖 Machine Learning

The project uses a Random Forest classifier trained on the historical dataset's environmental features.

The supplied dataset contains two original target labels:

- Moderate
- High

The classifier predicts these two labels. The separate rule-based scoring module generates the five project-defined risk categories.

The model uses a chronological train-test split to evaluate performance on a later period of historical data. Its metrics describe agreement with the dataset labels; they do not establish clinical accuracy or predict whether an individual will become ill.

## 🔮 Forecasting and Synthetic Data

The forecasting module supports experimental estimates using historical observations and a generated synthetic continuation.

- Historical observations end on **30 November 2024**.
- Synthetic continuation rows are generated estimates, not recorded weather observations.
- Future-date outputs are exploratory scenarios, not official weather forecasts.
- Forecasts more than approximately 30 days beyond the continuation are labelled illustrative scenarios.
- Synthetic demonstration scenarios are separate from the historical dataset and are not used to train the Random Forest.
- Hourly profiles are simulations because the source dataset contains daily rather than hourly observations.

The original CSV is not modified by the application.

## 🧪 Testing

The project includes tests covering core application components, including:

- Data loading and validation
- Risk-score calculations
- Random Forest classification
- Forecasting
- Hourly simulation
- Synthetic scenario generation
- Application behaviour

Run the suite with:

```bash
python -m pytest -q
```

## 🔭 Future Scope

Potential improvements include:

- Integration with live weather and air-quality APIs
- More advanced time-series forecasting
- Additional historical observations and external validation
- Geographic risk visualization
- Automated alerts and notifications
- Database integration for persistent data storage
- Broader model testing and rolling-origin evaluation

## ⚠️ Disclaimer

This project is developed for educational and academic purposes. Its risk scores, advisories, synthetic data, and forecasts are not medical diagnoses or official environmental warnings.

For real-world decisions, consult official guidance from the India Meteorological Department (IMD), the Central Pollution Control Board (CPCB), and relevant local health authorities.

## 📄 License

No license is specified by default. Add an appropriate `LICENSE` file if you want to define the terms under which others may use, modify, or distribute this project.


