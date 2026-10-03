# 📈 Predictive Analytics & Trend Forecasting Dashboard

A cloud-deployed machine learning application that ingests historical sales metrics, applies an automated data modeling pipeline using **Linear Regression**, and projects future operational demand via an interactive **Streamlit dashboard**.

## 📊 Project Objectives & Deliverables
* **Algorithmic Trend Forecasting:** Implements supervised learning (`Linear Regression` via `scikit-learn`) to establish a continuous mathematical baseline over historical data coordinates and compute statistical demand paths.
* **Accuracy Evaluation Matrix:** Calculates and displays real-time prediction accuracy scoring (**R² Score**) alongside absolute data points to evaluate model validity instantly.
* **Dynamic Projections Engine:** Integrates user-controlled sidebar parameters allowing corporate reviewers to simulate future timelines from 3 to 12 months ahead on the fly.

## 🛠️ System Frameworks Used
* **Machine Learning & Math Pipeline:** Python 3, scikit-learn, NumPy, Pandas
* **Data Visualization Engine:** Plotly Graph Objects (Multi-line Time-Series Mapping)
* **Web UI Framework:** Streamlit Cloud Architecture

## 🚀 Repository Blueprint
* `predictive_app.py` - Core application layer handling algorithmic model training, trend forecasting logic, and UI asset rendering.
* `historical_sales.csv` - Underlying structured time-series historical data catalog.
* `requirements.txt` - Deployment dependency manifest specifying precise environment installation parameters.
