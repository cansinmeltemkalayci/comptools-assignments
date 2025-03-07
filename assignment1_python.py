import pandas as pd
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Create the outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load the dataset directly from the URL
df = pd.read_csv('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64')

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')

# Drop rows with missing dates
df_cleaned = df_cleaned.dropna(subset=['sasdate'])

print(f"Dataset has {len(df_cleaned)} observations from {df_cleaned['sasdate'].min()} to {df_cleaned['sasdate'].max()}")

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

## transformation_codes contains the transformation codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$

# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

# After applying transformations, drop first 2 rows as they can have NaN due to differencing
df_cleaned = df_cleaned.iloc[2:].reset_index(drop=True)

# Check for variables with too many missing values
missing_data = df_cleaned.isnull().sum() / len(df_cleaned) * 100
print("\nPercentage of missing values in each variable:")
print(missing_data[missing_data > 10].sort_values(ascending=False))

# Select only the columns we need to reduce complexity
needed_columns_model1 = ['sasdate', 'INDPRO', 'CPIAUCSL', 'TB3MS']
needed_columns_model2 = ['sasdate', 'INDPRO', 'ACOGNO', 'BUSLOANS']

# Check if needed variables exist
available_columns_model1 = [col for col in needed_columns_model1 if col in df_cleaned.columns]
missing_columns_model1 = [col for col in needed_columns_model1 if col not in df_cleaned.columns]

available_columns_model2 = [col for col in needed_columns_model2 if col in df_cleaned.columns]
missing_columns_model2 = [col for col in needed_columns_model2 if col not in df_cleaned.columns]

if missing_columns_model1:
    print(f"\nWarning: The following needed columns for Model 1 are missing: {missing_columns_model1}")
else:
    print("\nAll needed columns for Model 1 are available.")

if missing_columns_model2:
    print(f"\nWarning: The following needed columns for Model 2 are missing: {missing_columns_model2}")
else:
    print("\nAll needed columns for Model 2 are available.")

print(f"\nFinal dataset has {len(df_cleaned)} rows")

############################################################################################################
## Plot transformed series for Model 1 (INDPRO, CPIAUCSL, TB3MS)
############################################################################################################
series_to_plot_model1 = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names_model1 = ['Industrial Production',
                'Inflation (CPI)',
                '3-month Treasury Bill rate']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot_model1), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot_model1, series_names_model1):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.savefig('outputs/transformed_series_model1.png')
plt.close()

############################################################################################################
## Plot transformed series for Model 2 (INDPRO, ACOGNO, BUSLOANS)
############################################################################################################
series_to_plot_model2 = ['INDPRO', 'ACOGNO', 'BUSLOANS']
series_names_model2 = ['Industrial Production',
                'Real Value of Manufacturers\' New Orders for Consumer Goods',
                'Commercial and Industrial Loans']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot_model2), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot_model2, series_names_model2):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.savefig('outputs/transformed_series_model2.png')
plt.close()

############################################################################################################
## Define forecast function for both models
############################################################################################################

def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='1999-12-01', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    """
    Calculate forecasts for target variable at different horizons.
    
    Parameters:
    -----------
    df_cleaned : pandas.DataFrame
        The cleaned and transformed dataset
    p : int
        Number of lags to include in the model
    H : list
        List of forecast horizons
    end_date : str
        End date for the estimation sample
    target : str
        Target variable to forecast
    xvars : list
        List of predictor variables
        
    Returns:
    --------
    tuple
        (forecast_errors, actual_values, forecasts)
    """
    # Convert end_date to pandas Timestamp if it's a string
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    # Subset df_cleaned to use only data up to end_date
    rt_df = df_cleaned[df_cleaned['sasdate'] <= end_date].copy()
    
    # Check if target and predictors exist in the data
    missing_vars = []
    if target not in rt_df.columns:
        missing_vars.append(target)
    for var in xvars:
        if var not in rt_df.columns:
            missing_vars.append(var)
            
    if missing_vars:
        print(f"Warning: The following variables are missing from the dataset: {missing_vars}")
        print(f"Available columns: {rt_df.columns.tolist()}")
        # Return NaN values if variables are missing
        return np.array([np.nan] * len(H)), np.array([np.nan] * len(H)), np.array([np.nan] * len(H))
    
    # Get the actual values of target at different steps ahead
    Y_actual = []
    for h in H:
        os = end_date + pd.DateOffset(months=h)
        # Find the closest date if exact date doesn't exist
        future_dates = df_cleaned['sasdate'][df_cleaned['sasdate'] >= os]
        if len(future_dates) > 0:
            closest_date = future_dates.min()
            actual_value = df_cleaned[df_cleaned['sasdate'] == closest_date][target].values
            if len(actual_value) > 0:
                Y_actual.append(actual_value[0] * 100)
            else:
                Y_actual.append(np.nan)
        else:
            Y_actual.append(np.nan)
    
    Yraw = rt_df[target]
    Xraw = rt_df[xvars]
    
    X = pd.DataFrame()
    # Add the lagged values of Y
    for lag in range(0, p + 1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)
    
    # Add the lagged values of predictors
    for col in Xraw.columns:
        for lag in range(0, p + 1):
            # Shift each column in the DataFrame and name it with a lag suffix
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    
    # Add a column of ones (for the intercept)
    X.insert(0, 'Ones', np.ones(len(X)))
    
    # Save last row of X (converted to numpy)
    X_T = X.iloc[-1:].values
    
    # Calculate forecasts for different horizons
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        # Subset getting only rows of X and y from p+1 to h-1
        y = y_h.iloc[p:-h].values if h > 0 and len(y_h) > p+h else y_h.iloc[p:].values
        X_ = X.iloc[p:-h].values if h > 0 and len(X) > p+h else X.iloc[p:].values
        
        # Filter out rows with NaN values
        mask = ~np.isnan(X_).any(axis=1) & ~np.isnan(y)
        X_ = X_[mask]
        y = y[mask]
        
        # Handle empty arrays
        if len(y) == 0 or len(X_) == 0:
            Yhat.append(np.nan)
            continue
            
        # Solving for the OLS estimator beta: (X'X)^{-1} X'Y
        try:
            beta_ols = solve(X_.T @ X_, X_.T @ y)
            # Produce the forecast (% change month-to-month of target)
            forecast = (X_T @ beta_ols)[0] * 100
            Yhat.append(forecast)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            Yhat.append(np.nan)
        except Exception:
            Yhat.append(np.nan)
    
    # Calculate the forecasting errors
    errors = np.array(Y_actual) - np.array(Yhat)
    
    return errors, np.array(Y_actual), np.array(Yhat)

############################################################################################################
## Perform real-time evaluation for both models
############################################################################################################

# We'll use a longer evaluation period for better comparison
start_date = pd.Timestamp('1999-12-01')
end_date = pd.Timestamp('2005-12-01')  # Evaluate over 6 years
H = [1, 4, 8]  # Forecast horizons: 1, 4, and 8 months ahead

# Initialize lists to store results for both models
errors_model1 = []
errors_model2 = []
dates = []
actuals = []
forecasts_model1 = []
forecasts_model2 = []

# Generate sequence of dates for evaluation
evaluation_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
print(f"Evaluating forecasts for {len(evaluation_dates)} dates from {start_date} to {end_date}")

# Perform real-time evaluation
for i, t0 in enumerate(evaluation_dates):
    # Add a debug print to track progress
    if i % 10 == 0:
        print(f"Processing {i+1}/{len(evaluation_dates)}: {t0}")
    
    # Skip dates with insufficient future data for evaluating long-horizon forecasts
    if t0 + pd.DateOffset(months=max(H)) > df_cleaned['sasdate'].max():
        continue
    
    # Calculate forecasts and errors for Model 1 (CPIAUCSL, TB3MS)
    error1, actual, forecast1 = calculate_forecast(df_cleaned, p=4, H=H, end_date=t0, 
                                                  target='INDPRO', xvars=['CPIAUCSL', 'TB3MS'])
    
    # Calculate forecasts and errors for Model 2 (ACOGNO, BUSLOANS)
    error2, _, forecast2 = calculate_forecast(df_cleaned, p=4, H=H, end_date=t0, 
                                             target='INDPRO', xvars=['ACOGNO', 'BUSLOANS'])
    
    # Store results
    errors_model1.append(error1)
    errors_model2.append(error2)
    dates.append(t0)
    actuals.append(actual)
    forecasts_model1.append(forecast1)
    forecasts_model2.append(forecast2)

# Convert lists to DataFrames for easier analysis
errors_df_model1 = pd.DataFrame(errors_model1, index=dates, columns=[f'h{h}' for h in H])
errors_df_model2 = pd.DataFrame(errors_model2, index=dates, columns=[f'h{h}' for h in H])
actuals_df = pd.DataFrame(actuals, index=dates, columns=[f'h{h}' for h in H])
forecasts_df_model1 = pd.DataFrame(forecasts_model1, index=dates, columns=[f'h{h}' for h in H])
forecasts_df_model2 = pd.DataFrame(forecasts_model2, index=dates, columns=[f'h{h}' for h in H])

# Calculate MSFE for each forecast horizon and model
msfe_model1 = errors_df_model1.apply(lambda x: (x**2).mean(skipna=True))
msfe_model2 = errors_df_model2.apply(lambda x: (x**2).mean(skipna=True))
rmsfe_model1 = np.sqrt(msfe_model1)
rmsfe_model2 = np.sqrt(msfe_model2)

print("\nRoot Mean Squared Forecast Error (RMSFE) for Model 1 (CPIAUCSL, TB3MS):")
for h_idx, h in enumerate(H):
    if np.isnan(rmsfe_model1[f'h{h}']):
        print(f"Horizon {h} month(s): No valid forecasts available")
    else:
        print(f"Horizon {h} month(s): {rmsfe_model1[f'h{h}']:.6f}")

print("\nRoot Mean Squared Forecast Error (RMSFE) for Model 2 (ACOGNO, BUSLOANS):")
for h_idx, h in enumerate(H):
    if np.isnan(rmsfe_model2[f'h{h}']):
        print(f"Horizon {h} month(s): No valid forecasts available")
    else:
        print(f"Horizon {h} month(s): {rmsfe_model2[f'h{h}']:.6f}")

############################################################################################################
## Plot actual vs forecast for both models (h=1)
############################################################################################################

plt.figure(figsize=(12, 6))
plt.plot(actuals_df.index, actuals_df['h1'], label='Actual', color='black', linewidth=2)
plt.plot(forecasts_df_model1.index, forecasts_df_model1['h1'], label='Model 1 Forecast', color='blue', linestyle='--')
plt.plot(forecasts_df_model2.index, forecasts_df_model2['h1'], label='Model 2 Forecast', color='red', linestyle='-.')
plt.title('1-Month Ahead Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Monthly % Change')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/forecast_vs_actual_h1_comparison.png')
plt.close()

############################################################################################################
## Plot forecast errors over time for both models
############################################################################################################

plt.figure(figsize=(12, 12))
for h_idx, h in enumerate(H):
    plt.subplot(len(H), 1, h_idx + 1)
    plt.plot(errors_df_model1.index, errors_df_model1[f'h{h}'], label=f'Model 1 (h={h})', color='blue')
    plt.plot(errors_df_model2.index, errors_df_model2[f'h{h}'], label=f'Model 2 (h={h})', color='red')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'Forecast Errors for h={h}')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/forecast_errors_comparison.png')
plt.close()

############################################################################################################
## Compare models with bar chart
############################################################################################################

# Create a bar chart comparing RMSFE for both models
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(H))

plt.bar(x - bar_width/2, [rmsfe_model1[f'h{h}'] for h in H], bar_width, label='Model 1 (CPIAUCSL, TB3MS)', color='blue', alpha=0.7)
plt.bar(x + bar_width/2, [rmsfe_model2[f'h{h}'] for h in H], bar_width, label='Model 2 (ACOGNO, BUSLOANS)', color='red', alpha=0.7)

plt.xlabel('Forecast Horizon')
plt.ylabel('RMSFE')
plt.title('RMSFE Comparison Between Models')
plt.xticks(x, [f'{h} Month(s)' for h in H])
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/rmsfe_comparison.png')
plt.close()

############################################################################################################
## Additional Analysis: Compare different lag specifications for both models
############################################################################################################

lag_specs = [1, 2, 4, 6, 12]
rmsfe_by_lag_model1 = {}
rmsfe_by_lag_model2 = {}

for p in lag_specs:
    errors_p_model1 = []
    errors_p_model2 = []
    
    # Use a subset of dates to speed up computation
    for t0 in evaluation_dates[:50]:  # Reduced from all dates to 50 for faster testing
        if t0 + pd.DateOffset(months=max(H)) > df_cleaned['sasdate'].max():
            continue
        
        error1, _, _ = calculate_forecast(df_cleaned, p=p, H=H, end_date=t0, target='INDPRO', xvars=['CPIAUCSL', 'TB3MS'])
        error2, _, _ = calculate_forecast(df_cleaned, p=p, H=H, end_date=t0, target='INDPRO', xvars=['ACOGNO', 'BUSLOANS'])
        
        errors_p_model1.append(error1)
        errors_p_model2.append(error2)
    
    errors_p_df_model1 = pd.DataFrame(errors_p_model1, columns=[f'h{h}' for h in H])
    errors_p_df_model2 = pd.DataFrame(errors_p_model2, columns=[f'h{h}' for h in H])
    
    msfe_p_model1 = errors_p_df_model1.apply(lambda x: (x**2).mean(skipna=True))
    msfe_p_model2 = errors_p_df_model2.apply(lambda x: (x**2).mean(skipna=True))
    
    rmsfe_p_model1 = np.sqrt(msfe_p_model1)
    rmsfe_p_model2 = np.sqrt(msfe_p_model2)
    
    rmsfe_by_lag_model1[f'p={p}'] = rmsfe_p_model1
    rmsfe_by_lag_model2[f'p={p}'] = rmsfe_p_model2

# Convert to DataFrames for easier analysis
rmsfe_by_lag_df_model1 = pd.DataFrame(rmsfe_by_lag_model1)
rmsfe_by_lag_df_model2 = pd.DataFrame(rmsfe_by_lag_model2)

print("\nRMSFE by Lag Specification for Model 1:")
print(rmsfe_by_lag_df_model1)

print("\nRMSFE by Lag Specification for Model 2:")
print(rmsfe_by_lag_df_model2)

# Plot RMSFE by lag specification for Model 1
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
for h_idx, h in enumerate(H):
    plt.plot(lag_specs, [rmsfe_by_lag_df_model1.loc[f'h{h}', f'p={p}'] for p in lag_specs], 
             marker='o', label=f'h={h}')
plt.title('Model 1: RMSFE by Lag Specification')
plt.xlabel('Number of Lags (p)')
plt.ylabel('RMSFE')
plt.legend()
plt.grid(True)
plt.xticks(lag_specs)

# Plot RMSFE by lag specification for Model 2
plt.subplot(2, 1, 2)
for h_idx, h in enumerate(H):
    plt.plot(lag_specs, [rmsfe_by_lag_df_model2.loc[f'h{h}', f'p={p}'] for p in lag_specs], 
             marker='o', label=f'h={h}')
plt.title('Model 2: RMSFE by Lag Specification')
plt.xlabel('Number of Lags (p)')
plt.ylabel('RMSFE')
plt.legend()
plt.grid(True)
plt.xticks(lag_specs)

plt.tight_layout()
plt.savefig('outputs/rmsfe_by_lag_comparison.png')
plt.close()

############################################################################################################
## Save results to CSV files in outputs directory
############################################################################################################

errors_df_model1.to_csv('outputs/forecast_errors_model1.csv')
errors_df_model2.to_csv('outputs/forecast_errors_model2.csv')
actuals_df.to_csv('outputs/actual_values.csv')
forecasts_df_model1.to_csv('outputs/forecasts_model1.csv')
forecasts_df_model2.to_csv('outputs/forecasts_model2.csv')
pd.DataFrame({'RMSFE_Model1': rmsfe_model1, 'RMSFE_Model2': rmsfe_model2}).to_csv('outputs/rmsfe_comparison.csv')
rmsfe_by_lag_df_model1.to_csv('outputs/rmsfe_by_lag_model1.csv')
rmsfe_by_lag_df_model2.to_csv('outputs/rmsfe_by_lag_model2.csv')

############################################################################################################
## Print summary of results
############################################################################################################

print("\nReal-time Evaluation Results Summary:")
print(f"Evaluation period: {dates[0]} to {dates[-1]}")
print(f"Number of forecasts evaluated: {len(dates)}")

print("\nRMSFE for Model 1 (CPIAUCSL, TB3MS):")
for h_idx, h in enumerate(H):
    if np.isnan(rmsfe_model1[f'h{h}']):
        print(f"  Horizon {h} month(s): No valid forecasts available")
    else:
        print(f"  Horizon {h} month(s): {rmsfe_model1[f'h{h}']:.6f}")

print("\nRMSFE for Model 2 (ACOGNO, BUSLOANS):")
for h_idx, h in enumerate(H):
    if np.isnan(rmsfe_model2[f'h{h}']):
        print(f"  Horizon {h} month(s): No valid forecasts available")
    else:
        print(f"  Horizon {h} month(s): {rmsfe_model2[f'h{h}']:.6f}")

print("\nComparison of Models:")
for h_idx, h in enumerate(H):
    rmsfe1 = rmsfe_model1[f'h{h}']
    rmsfe2 = rmsfe_model2[f'h{h}']
    if not (np.isnan(rmsfe1) or np.isnan(rmsfe2)):
        better_model = "Model 1" if rmsfe1 < rmsfe2 else "Model 2"
        improvement = abs(rmsfe1 - rmsfe2) / max(rmsfe1, rmsfe2) * 100
        print(f"  Horizon {h} month(s): {better_model} performs better by {improvement:.2f}%")
    else:
        print(f"  Horizon {h} month(s): Cannot compare (missing values)")

print("\nBest lag specification for Model 1:")
for h_idx, h in enumerate(H):
    # Check for NaN values
    if rmsfe_by_lag_df_model1.loc[f'h{h}'].isna().all():
        print(f"  Horizon {h} month(s): No valid model found (all values are NaN)")
    else:
        # Drop NaN values before finding minimum
        valid_lags = rmsfe_by_lag_df_model1.loc[f'h{h}'].dropna()
        if len(valid_lags) > 0:
            best_lag = valid_lags.idxmin()
            best_rmsfe = valid_lags[best_lag]
            print(f"  Horizon {h} month(s): {best_lag} (RMSFE: {best_rmsfe:.6f})")
        else:
            print(f"  Horizon {h} month(s): No valid model found")

print("\nBest lag specification for Model 2:")
for h_idx, h in enumerate(H):
    # Check for NaN values
    if rmsfe_by_lag_df_model2.loc[f'h{h}'].isna().all():
        print(f"  Horizon {h} month(s): No valid model found (all values are NaN)")
    else:
        # Drop NaN values before finding minimum
        valid_lags = rmsfe_by_lag_df_model2.loc[f'h{h}'].dropna()
        if len(valid_lags) > 0:
            best_lag = valid_lags.idxmin()
            best_rmsfe = valid_lags[best_lag]
            print(f"  Horizon {h} month(s): {best_lag} (RMSFE: {best_rmsfe:.6f})")
        else:
            print(f"  Horizon {h} month(s): No valid model found")
