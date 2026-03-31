# AI-Based Sleep Recommendation System

An end-to-end machine learning project that predicts daily sleep duration from lifestyle habits and provides personalized, rule-based recommendations.

This repository includes:
- A reproducible model training workflow in Jupyter Notebook
- A deployed Streamlit application for interactive predictions
- Saved model artifacts for inference (`model.pkl`, `scaler.pkl`)

## Problem Statement

Sleep quality and duration are strongly influenced by day-to-day behavior. This project predicts expected sleep duration (in hours) using lifestyle inputs and gives users practical steps to improve their routine.

## Features

- Predicts sleep duration using a trained Linear Regression model
- Applies the same StandardScaler used during training before inference
- Generates personalized recommendations based on user input thresholds
- Validates daily time budget (tracked activities cannot exceed 24 hours)
- Displays lifestyle analysis summary and actionable recommendations

## Input Features

- WorkoutTime (hours/day)
- ReadingTime (hours/day)
- PhoneTime (hours/day)
- WorkHours (hours/day)
- CaffeineIntake (mg/day)
- RelaxationTime (hours/day)

## Target Variable

- SleepTime (hours)

## Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- Matplotlib, Seaborn
- XGBoost (used in model comparison)

## Repository Structure

```text
.
|-- app_ui.py                     # Streamlit UI (primary app)
|-- app.py                        # Flask API variant
|-- sleep_model_training.ipynb    # Data prep, training, evaluation, artifact export
|-- model.pkl                     # Trained best model artifact
|-- scaler.pkl                    # Trained feature scaler artifact
|-- readme.md
```

## Training Pipeline

The notebook workflow in `sleep_model_training.ipynb` follows this structure:

1. Load dataset
2. Basic exploration and data checks
3. Outlier filtering on target sleep duration
4. Feature/target separation
5. Train-test split
6. Standardization using training features only (no data leakage)
7. Multi-model comparison (Linear Regression, Random Forest, Gradient Boosting, SVR, AdaBoost, XGBoost)
8. Select best model using MSE and R²
9. Export artifacts:
	 - `model.pkl`
	 - `scaler.pkl`

## Recommendation Engine (Rule-Based)

The Streamlit app includes simple, transparent business rules (not ML-generated advice), for example:

- High phone usage threshold checks
- High caffeine intake threshold checks
- Excessive work hours checks
- Low relaxation / workout / reading checks

Recommendations are:
- Personalized using actual user input values
- Actionable with suggested target values
- Combined into multiple bullet points when several habits need improvement

## Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd sleep-time-prediction
```

### 2. Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn xgboost kagglehub flask
```

## Run the App

### Streamlit (recommended)

```bash
python -m streamlit run app_ui.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### Flask API (optional)

```bash
python app.py
```

## Inference Consistency

To ensure reliable predictions, inference uses the same preprocessing as training:

1. User input is arranged in the trained feature order
2. Input is transformed using `scaler.pkl`
3. Prediction is generated using `model.pkl`
4. Prediction is bounded to a realistic range [0, 12] hours in the UI

## Validation Rules in UI

- Realistic upper bounds for each input field
- Tracked daily time validation:
	- `total_time = workout + reading + phone + work + relaxation`
	- If `total_time > 24`, prediction is blocked and an error is shown
	- If near 24, a warning is shown

## Evaluation Metrics

Models are compared primarily using:

- Mean Squared Error (MSE)
- R² Score

Linear Regression is currently selected as the production model artifact.

## Troubleshooting

### App predicts 0 hours repeatedly

Likely cause: raw input sent directly to model without scaling.

Fix:
- Ensure both `model.pkl` and `scaler.pkl` exist
- Ensure app applies `scaler.transform(...)` before `model.predict(...)`

### Streamlit command fails

Use:

```bash
python -m streamlit run app_ui.py
```

Also confirm packages are installed in the active environment.

## Future Enhancements

- Add feature importance/explainability views
- Add confidence intervals for predictions
- Track weekly user trends and progress
- Add model versioning and experiment tracking
- Containerize deployment with Docker

## Author

Sanjam

