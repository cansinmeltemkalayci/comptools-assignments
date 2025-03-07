# Computational Tools for Macroeconometrics - Assignment 1

## Overview
This repository contains the code for Assignment 1 of the Computational Tools for Macroeconometrics course. The assignment focuses on forecasting macroeconomic indicators using the FRED-MD dataset, applying time series transformations, and evaluating forecasting performance through real-time evaluation.

## Files
- `assignment1_python.py`: Python implementation of the assignment
- `assignment1_r.r`: R implementation of the assignment (starter code)
- `assignment1_julia.jl`: Julia implementation of the assignment (starter code)
- `comptools_ass1.qmd`: Assignment description in Quarto format
- `comptools_ass1.pdf`: Assignment description in PDF format
- `comptools_ass1.html`: Assignment description in HTML format
- `STEPS_TR.md`: Step-by-step guide in Turkish
- `STEPS_EN.md`: Step-by-step guide in English
- `requirements.txt`: Required Python packages

## What the Code Does
The Python implementation (`assignment1_python.py`) performs the following tasks:

1. **Data Loading and Cleaning**:
   - Downloads the FRED-MD dataset directly from the Federal Reserve Bank of St. Louis.
   - Cleans the data and applies appropriate transformations based on transformation codes.

2. **Data Visualization**:
   - Visualizes key economic indicators for two different models:
     - Model 1: Industrial Production, Inflation, and 3-month Treasury Bill rate
     - Model 2: Industrial Production, Manufacturers' New Orders, and Commercial Loans

3. **ARX Model Implementation**:
   - Implements autoregressive models with exogenous variables (ARX).
   - Creates design matrices with lagged values of target and predictor variables.

4. **Real-time Evaluation**:
   - Performs real-time forecasting evaluation from 1999 to 2005.
   - Generates forecasts at multiple horizons (1, 4, and 8 months ahead).
   - Compares two different models:
     - Model 1 uses CPIAUCSL and TB3MS as predictors
     - Model 2 uses ACOGNO and BUSLOANS as predictors

5. **Forecast Evaluation**:
   - Calculates Mean Squared Forecast Error (MSFE) and Root Mean Squared Forecast Error (RMSFE).
   - Compares forecast performance across different forecast horizons.
   - Evaluates which model performs better and by how much.

6. **Model Comparison**:
   - Evaluates how different lag specifications (1, 2, 4, 6, and 12 lags) affect forecast performance.
   - Identifies the optimal lag structure for each forecast horizon and each model.

7. **Results Visualization**:
   - Generates plots comparing actual values vs. forecasts for both models.
   - Visualizes forecast errors over time.
   - Creates plots showing how RMSFE varies with different lag specifications.
   - Creates comparative bar charts for model performance evaluation.

## Running the Code
To run the Python implementation:

1. **Create and activate a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install requirements**:
```bash
pip install -r requirements.txt
```

3. **Run the code**:
```bash
python assignment1_python.py
```

The code will:
1. Download and process the FRED-MD dataset
2. Perform the real-time evaluation with two different models
3. Generate performance metrics and visualization plots
4. Save results to CSV files and images in the `outputs` directory

## Output Files
The code generates the following output files in the `outputs` directory:

- `transformed_series_model1.png`: Plot of transformed variables for Model 1
- `transformed_series_model2.png`: Plot of transformed variables for Model 2
- `forecast_vs_actual_h1_comparison.png`: Plot comparing both models' forecasts vs. actual values
- `forecast_errors_comparison.png`: Plot showing forecast errors over time for both models
- `rmsfe_comparison.png`: Bar chart comparing RMSFE for both models
- `rmsfe_by_lag_comparison.png`: Plot showing RMSFE by lag specification for both models
- Various CSV files with detailed results

## Results Summary
- Model 1 (CPIAUCSL, TB3MS) outperformed Model 2 (ACOGNO, BUSLOANS) in all forecast horizons.
- The most significant difference was observed in the 8-month forecast horizon (14.51% better).
- Different lag structures were found to be optimal for different forecast horizons and models.

## Requirements
- Python 3.6+
- pandas
- numpy
- matplotlib

## Step-by-Step Process
For a detailed step-by-step explanation of the analysis process, see:
- [STEPS_EN.md](STEPS_EN.md) (English)
- [STEPS_TR.md](STEPS_TR.md) (Turkish)
