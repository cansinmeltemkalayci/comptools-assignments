#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 4: Lasso and Sparse Linear Model Recovery
Computational Tools for Macroeconometrics

This assignment examines the effectiveness of the Lasso (Least Absolute Shrinkage and Selection Operator) 
method in variable selection and coefficient estimation in sparse linear models. 
It also includes a comparison with Ridge regression.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.rcParams['figure.figsize'] = (10, 6)

#############################################################
# 1. Data Generating Process (DGP)
#############################################################
print("1. Data Generating Process (DGP)")
print("-" * 50)

# Set parameters for the data generating process
n_samples = 200        # Number of observations
n_features = 50        # Total number of features
n_informative = 10     # Number of features with non-zero coefficients
noise_level = 1.0      # Standard deviation of the noise

# Generate feature matrix X
# Each column is a feature drawn from standard normal distribution
X = np.random.randn(n_samples, n_features)

# Create the true coefficient vector (beta)
# Most coefficients are zero (sparse model)
true_coefficients = np.zeros(n_features)

# Randomly select which features will have non-zero coefficients
informative_features = np.random.choice(n_features, n_informative, replace=False)
print(f"True informative features indices: {sorted(informative_features)}")

# Assign non-zero values to selected coefficients
# Values are drawn from a normal distribution with larger variance
for idx in informative_features:
    true_coefficients[idx] = np.random.randn() * 3

# Generate the response variable Y
# Y = X * beta + noise
Y = X @ true_coefficients + np.random.randn(n_samples) * noise_level

# Save the data and true coefficients for later analysis
data_dict = {
    'X': X,
    'Y': Y,
    'true_coefficients': true_coefficients,
    'informative_features': informative_features
}

# Create a DataFrame to better visualize the coefficients
coef_df = pd.DataFrame({
    'feature_index': range(n_features),
    'true_coefficient': true_coefficients
})

# Show the non-zero coefficients
print("\nNon-zero coefficients:")
print(coef_df[coef_df['true_coefficient'] != 0])

#############################################################
# 2. Train-Test Split
#############################################################
print("\n2. Train-Test Split")
print("-" * 50)

# Split the data into training and testing sets
# We use 70% for training and 30% for testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Standardize the features (important for regularized regression)
# Fit the scaler on training data and apply to both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#############################################################
# 3. Implementing Lasso Regression
#############################################################
print("\n3. Implementing Lasso Regression")
print("-" * 50)
print("Lasso uses the L1 norm to set some coefficients exactly to zero.")
print("This automatically performs feature selection.")

# Define different alpha values to test
# Alpha controls the strength of regularization
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

# Store results for each alpha
lasso_results = {}

for alpha in alphas:
    # Create and fit Lasso model
    # max_iter: maximum number of iterations for optimization
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, Y_train)
    
    # Make predictions
    Y_train_pred = lasso.predict(X_train_scaled)
    Y_test_pred = lasso.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(lasso.coef_ != 0)
    
    # Store results
    lasso_results[alpha] = {
        'model': lasso,
        'coefficients': lasso.coef_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_nonzero_coef': n_nonzero
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Non-zero coefficients: {n_nonzero}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

#############################################################
# 4. Implementing Ridge Regression
#############################################################
print("\n4. Implementing Ridge Regression")
print("-" * 50)
print("Ridge uses the L2 norm to shrink coefficients towards zero,")
print("but it typically doesn't set them exactly to zero.")

# Store Ridge results for comparison
ridge_results = {}

for alpha in alphas:
    # Create and fit Ridge model
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, Y_train)
    
    # Make predictions
    Y_train_pred = ridge.predict(X_train_scaled)
    Y_test_pred = ridge.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    
    # For Ridge, count "effectively zero" coefficients (very small)
    threshold = 0.001
    n_small = np.sum(np.abs(ridge.coef_) < threshold)
    
    # Store results
    ridge_results[alpha] = {
        'model': ridge,
        'coefficients': ridge.coef_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_small_coef': n_small
    }
    
    print(f"\nAlpha = {alpha}")
    print(f"  Coefficients < {threshold}: {n_small}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

#############################################################
# 5. Visualizing Coefficient Recovery
#############################################################
print("\n5. Visualizing Coefficient Recovery")
print("-" * 50)
print("We visualize how well the Lasso and Ridge methods estimate the true coefficients.")

# Select a specific alpha for detailed comparison
selected_alpha = 0.1

# Get the coefficients for the selected alpha
lasso_coef = lasso_results[selected_alpha]['coefficients']
ridge_coef = ridge_results[selected_alpha]['coefficients']

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Lasso coefficients vs True coefficients
ax1 = axes[0, 0]
ax1.scatter(true_coefficients, lasso_coef, alpha=0.6)
ax1.plot([-5, 5], [-5, 5], 'r--', label='Perfect recovery')
ax1.set_xlabel('True Coefficients')
ax1.set_ylabel('Lasso Coefficients')
ax1.set_title(f'Lasso Coefficient Recovery (α={selected_alpha})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Ridge coefficients vs True coefficients
ax2 = axes[0, 1]
ax2.scatter(true_coefficients, ridge_coef, alpha=0.6)
ax2.plot([-5, 5], [-5, 5], 'r--', label='Perfect recovery')
ax2.set_xlabel('True Coefficients')
ax2.set_ylabel('Ridge Coefficients')
ax2.set_title(f'Ridge Coefficient Recovery (α={selected_alpha})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Coefficient path for Lasso
ax3 = axes[1, 0]
for idx in informative_features:
    coef_path = [lasso_results[alpha]['coefficients'][idx] for alpha in alphas]
    ax3.plot(alphas, coef_path, 'b-', linewidth=2, alpha=0.8)
# Plot non-informative features in lighter color
for idx in range(n_features):
    if idx not in informative_features:
        coef_path = [lasso_results[alpha]['coefficients'][idx] for alpha in alphas]
        ax3.plot(alphas, coef_path, 'gray', linewidth=0.5, alpha=0.3)
ax3.set_xscale('log')
ax3.set_xlabel('Alpha (log scale)')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Lasso Coefficient Path')
ax3.grid(True, alpha=0.3)

# Plot 4: Number of non-zero coefficients vs alpha
ax4 = axes[1, 1]
nonzero_counts = [lasso_results[alpha]['n_nonzero_coef'] for alpha in alphas]
ax4.plot(alphas, nonzero_counts, 'o-', linewidth=2, markersize=8)
ax4.axhline(y=n_informative, color='r', linestyle='--', 
            label=f'True number ({n_informative})')
ax4.set_xscale('log')
ax4.set_xlabel('Alpha (log scale)')
ax4.set_ylabel('Number of Non-zero Coefficients')
ax4.set_title('Sparsity vs Regularization Strength')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coefficient_recovery.png')
plt.show()

#############################################################
# 6. Cross-Validation for Optimal Alpha
#############################################################
print("\n6. Cross-Validation for Optimal Alpha")
print("-" * 50)
print("We find the best alpha value using K-fold cross-validation.")

# Define a wider range of alphas to test
cv_alphas = np.logspace(-4, 1, 20)

# Store the cross-validation scores
lasso_cv_scores = []
ridge_cv_scores = []

# Perform cross-validation for each alpha
for alpha in cv_alphas:
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso_scores = cross_val_score(lasso, X_train_scaled, Y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
    lasso_cv_scores.append(np.mean(-lasso_scores))  # Convert to positive MSE
    
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge_scores = cross_val_score(ridge, X_train_scaled, Y_train, 
                                 cv=5, scoring='neg_mean_squared_error')
    ridge_cv_scores.append(np.mean(-ridge_scores))  # Convert to positive MSE

# Find the best alphas
best_lasso_idx = np.argmin(lasso_cv_scores)
best_ridge_idx = np.argmin(ridge_cv_scores)
best_lasso_alpha = cv_alphas[best_lasso_idx]
best_ridge_alpha = cv_alphas[best_ridge_idx]

print(f"Best Lasso alpha: {best_lasso_alpha:.6f}")
print(f"Best Ridge alpha: {best_ridge_alpha:.6f}")

# Plot the cross-validation results
plt.figure(figsize=(12, 6))
plt.plot(cv_alphas, lasso_cv_scores, 'b-o', label='Lasso')
plt.plot(cv_alphas, ridge_cv_scores, 'r-o', label='Ridge')
plt.axvline(x=best_lasso_alpha, color='b', linestyle='--', 
           label=f'Best Lasso alpha: {best_lasso_alpha:.6f}')
plt.axvline(x=best_ridge_alpha, color='r', linestyle='--',
           label=f'Best Ridge alpha: {best_ridge_alpha:.6f}')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Cross-validation MSE')
plt.xscale('log')
plt.title('Cross-validation Error vs. Regularization Strength')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cross_validation.png')
plt.show()

# Train final models with optimal alpha
final_lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000).fit(X_train_scaled, Y_train)
final_ridge = Ridge(alpha=best_ridge_alpha).fit(X_train_scaled, Y_train)

# Get predictions
lasso_y_test_pred = final_lasso.predict(X_test_scaled)
ridge_y_test_pred = final_ridge.predict(X_test_scaled)

# Calculate test metrics
lasso_test_mse = mean_squared_error(Y_test, lasso_y_test_pred)
ridge_test_mse = mean_squared_error(Y_test, ridge_y_test_pred)
lasso_test_r2 = r2_score(Y_test, lasso_y_test_pred)
ridge_test_r2 = r2_score(Y_test, ridge_y_test_pred)

print("\nFinal Model Test Performance:")
print(f"  Lasso (alpha={best_lasso_alpha:.6f})")
print(f"    Test MSE: {lasso_test_mse:.4f}, Test R²: {lasso_test_r2:.4f}")
print(f"    Non-zero coefficients: {np.sum(final_lasso.coef_ != 0)} of {n_features}")
print(f"  Ridge (alpha={best_ridge_alpha:.6f})")
print(f"    Test MSE: {ridge_test_mse:.4f}, Test R²: {ridge_test_r2:.4f}")
print(f"    Small coefficients (< 0.001): {np.sum(np.abs(final_ridge.coef_) < 0.001)} of {n_features}")

#############################################################
# 7. Summary
#############################################################
print("\n7. Summary")
print("-" * 50)
print("In this study:")
print("1. We applied Lasso and Ridge regression methods to sparse datasets.")
print("2. We observed that Lasso can produce exactly zero coefficients, thus performing feature selection.")
print("3. We saw that Ridge shrinks coefficients towards zero, but typically doesn't set them exactly to zero.")
print("4. We found the optimal regularization strength for both models using cross-validation.")
print("5. We evaluated Lasso's ability to select the truly informative variables.")

