# Requirements traceability — aligned to the submitted OOSE SRS

This implementation keeps the submitted report and UML diagrams unchanged. The SRS describes the **core system** as historical environmental monitoring, risk-score/category display, Random Forest classification, historical trend visualization, and health advisories. Features explicitly marked *experimental extension* are additional demonstrations and are not presented as mandatory SRS requirements.

| ID | Requirement / scope | Implementation | Status / note |
|---|---|---|---|
| SRS-UI-1 | Browser-based dashboard | `app.py` (Streamlit) | Implemented; browser UI framework differs from the HTML/CSS/JavaScript design description in the report |
| SRS-HIST-1 | Load configured historical environmental dataset | `src/data_loader.py`, `data/delhi_heatwave_airquality.csv` | Implemented using local CSV; read-only, no live feed |
| SRS-HIST-2 | Navigate available historical date range and handle missing dates | Sidebar date range, Overview, Data Explorer | Implemented; missing-date selections show a message |
| SRS-RISK-1 | Display risk score, category and environmental measurements | `src/risk_scoring.py`, Overview, Risk Calculator | Implemented; five-level score categories are a project-defined extension of the four categories named in the report |
| SRS-ML-1 | Random Forest classification with expected feature schema | `src/risk_classifier.py`, ML Risk Classifier tab | Implemented; the supplied dataset contains only Moderate/High target labels, so the RF learns only those two labels |
| SRS-TREND-1 | Historical trends and charts | Trends, Overview | Implemented |
| SRS-ADV-1 | Advisory mapping by risk category | `src/advisories.py` | Implemented; general informational guidance, not medical advice |
| SRS-DATA-1 | Data validation and unavailable-data handling | `src/data_loader.py` | Implemented |
| EXT-1 | Five-level risk calculator | Risk Calculator | Optional extension; Very Low/Low/Moderate/High/Severe are score-rule categories, not five ML classes |
| EXT-2 | Future-date scenario estimates | `src/forecasting.py`, Forecast (Experimental) tab | Optional experimental extension; original rows preserved, synthetic continuation visually separated and labelled; output is not an official weather forecast |
| EXT-3 | Hourly risk profile | `src/hourly_analysis.py` | Optional simulation based on daily values and assumptions; dataset has no hourly observations |
| EXT-4 | Synthetic scenarios covering all five levels | `src/synthetic_data.py` | Optional demo/testing extension; synthetic records are labelled and not mixed into real data or model evaluation |
| EXT-5 | Data explorer and CSV downloads | Data Explorer and download controls | Optional usability extension |
| SRS-COMM-1 | Do not claim live weather/AQI API integration unless implemented | Local CSV data loader | No live API is implemented or claimed |
| SRS-DB-1 | Treat CSV as a structured data file, not as a database | `data/delhi_heatwave_airquality.csv` | No SQL database is implemented; historical file is read-only |

## UML and report handling

The submitted UML diagrams are intentionally not edited. Some diagrams depict a planned database, external weather/AQI API and administrator roles; this runnable implementation currently uses a local CSV, does not connect to a live API, and does not implement authentication or administrator CRUD. These are documented design-versus-implementation limitations, not features claimed as complete. The app's core workflow remains aligned with historical monitoring, risk assessment, Random Forest classification, trends and advisories.
