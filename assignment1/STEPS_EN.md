# Steps

## 1. Environment Setup

[+] Created a virtual environment.
```shell
python3 -m venv venv
```

[+] Prepared requirements.txt for required libraries.

[+] Activated the virtual environment.
```shell
source venv/bin/activate
```

[+] Installed required libraries.
```shell
pip install -r requirements.txt
```

## 2. Data Loading and Preparation

[+] Downloaded the FRED-MD data file.
```
https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv
```

[+] Read the data file using pandas.

[+] Extracted the transformation codes row and removed it from the DataFrame.

[+] Converted the 'sasdate' column to datetime format.

[+] Removed rows with missing date values.

## 3. Data Transformations

[+] Stored transformation codes in a data structure.

[+] Defined the apply_transformation() function:
  - 1: No transformation
  - 2: First difference (Δx_t)
  - 3: Second difference (Δ²x_t)
  - 4: Logarithm (log(x_t))
  - 5: First difference of logarithm (Δlog(x_t))
  - 6: Second difference of logarithm (Δ²log(x_t))
  - 7: Percentage change (x_t/x_{t-1} - 1)

[+] Applied appropriate transformations to all variables.

[+] Removed the first two rows with missing values caused by transformations.

## 4. Data Exploration and Visualization

[+] Calculated the percentage of missing values in each variable.

[+] Selected variables for Model 1: INDPRO, CPIAUCSL, TB3MS.
  - INDPRO: Industrial Production
  - CPIAUCSL: Inflation (CPI)
  - TB3MS: 3-month Treasury Bill rate

[+] Selected variables for Model 2: INDPRO, ACOGNO, BUSLOANS.
  - INDPRO: Industrial Production
  - ACOGNO: Real Value of Manufacturers' New Orders for Consumer Goods
  - BUSLOANS: Commercial and Industrial Loans

[+] Created time series plots of transformed variables for both Model 1 and Model 2.

## 5. Forecasting Models

[+] Developed the calculate_forecast() function to implement ARX models.

[+] The function performs the following operations:
  - Filters data up to a specific date
  - Gets actual values for specific horizons
  - Creates a design matrix containing lagged values of dependent and independent variables
  - Estimates model parameters using OLS
  - Calculates forecasts and forecast errors for different horizons

## 6. Real-time Evaluation

[+] Created evaluation dates for the period from 1999-12-01 to 2005-12-01.

[+] For each evaluation date:
  - Generated forecasts for 1, 4, and 8 months ahead using Model 1 and Model 2
  - Calculated forecast errors
  - Stored results in lists

[+] Converted list data to DataFrames.

[+] Calculated MSFE and RMSFE for each forecast horizon and model.

## 7. Results Visualization

[+] Created comparison plots of 1-month ahead forecasts from Model 1 and Model 2 with actual values.

[+] Created plots showing the evolution of forecast errors over time for all horizons.

[+] Created a bar chart comparing RMSFE values for the two models.

## 8. Comparing Different Model Structures

[+] Calculated RMSFE values for different lag structures (1, 2, 4, 6, 12) for both models and each forecast horizon.

[+] Created plots showing the performance of different lag structures.

[+] Determined the best lag structure for each forecast horizon.

## 9. Saving and Reporting Results

[+] Saved all results and plots to the outputs folder.

[+] Reported results as output:
  - RMSFE values for each model
  - Model comparisons (which model is better and by how much)
  - Best lag structures for each model

## 10. Conclusions and Findings

[+] Model 1 (CPIAUCSL, TB3MS) outperformed Model 2 (ACOGNO, BUSLOANS) in all forecast horizons.

[+] The most significant difference was observed in the 8-month forecast horizon (14.51% better).

[+] Best lag structures for Model 1:
  - p=1 for 1-month horizon
  - p=12 for 4-month horizon
  - p=6 for 8-month horizon

[+] Best lag structures for Model 2:
  - p=4 for 1-month horizon
  - p=1 for 4-month horizon
  - p=1 for 8-month horizon 