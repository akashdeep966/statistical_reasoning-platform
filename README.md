# 🧠 Statistical Reasoning & Analytics Platform

A comprehensive statistical analysis platform that behaves like a competent statistician rather than a generic AutoML tool.

## Features

- **Semantic Understanding**: Auto-classifies variables and guards against spurious correlations
- **Assumption-First Modeling**: Checks normality, homoscedasticity, and multicollinearity before training
- **Explainable AI**: Permutation feature importance with human-readable interpretations
- **Practical Significance**: Distinguishes statistical significance from business impact
- **Model Registry**: Downloadable deployment packages with metadata

## Deployed App

[Click here to view the live app](https://your-app-url.streamlit.app)

## Local Setup

```bash
pip install -r requirements.txt
streamlit run statistical_reasoning_platform.py
```

## Project Structure

```
├── statistical_reasoning_platform.py   # Main application
├── requirements.txt                     # Python dependencies
├── .streamlit/
│   └── config.toml                      # Streamlit UI configuration
└── README.md                            # This file
```

## Data Upload

The app supports:
- CSV files
- Excel files (.xlsx)
- TSV files

If no file is uploaded, it automatically loads a built-in diabetes dataset for demonstration.

## Methodology

The platform follows a rigorous 7-stage pipeline:

1. **Dataset Understanding** - Data inventory and quality metrics
2. **Semantic Variable Understanding** - Domain-aware variable classification
3. **Data Quality Assessment** - Missing values, outliers, imputation recommendations
4. **Exploratory Data Analysis** - Univariate, bivariate, and multivariate analysis
5. **Assumption Engine** - Normality, homoscedasticity, multicollinearity tests
6. **Model Selection & Training** - Multiple algorithms with cross-validation
7. **Interpretation & Deployment** - Feature importance, practical significance, model registry

## License

MIT License
