import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Tuple, List, Dict
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

def simulate_ar1(n: int, phi: float, sigma: float) -> np.ndarray:
    """
    Simulate an AR(1) process.
    
    Parameters:
    -----------
    n : int
        Number of observations
    phi : float
        Coefficient of AR(1) process
    sigma : float
        Standard deviation of the innovation term
    
    Returns:
    --------
    np.ndarray
        Simulated AR(1) process
    """
    errors = np.zeros(n)
    eta = np.random.normal(0, sigma, n)  # white noise
    for t in range(1, n):
        errors[t] = phi * errors[t - 1] + eta[t]
    return errors

def simulate_regression_with_ar1_errors(n: int, beta0: float, beta1: float, 
                                     phi_x: float, phi_u: float, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a regression model with AR(1) error terms.
    
    Parameters:
    -----------
    n : int
        Number of observations
    beta0 : float
        Intercept of the regression model
    beta1 : float
        Slope of the regression model
    phi_x : float
        AR(1) coefficient for x
    phi_u : float
        AR(1) coefficient for the errors
    sigma : float
        Standard deviation of the white noise
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        x (independent variable), y (dependent variable), errors (AR(1) process)
    """
    x = simulate_ar1(n, phi_x, sigma)
    u = simulate_ar1(n, phi_u, sigma)
    y = beta0 + beta1 * x + u
    return x, y, u

def moving_block_bootstrap(x: np.ndarray, y: np.ndarray, block_length: int, 
                         num_bootstrap: int) -> np.ndarray:
    """
    Perform moving block bootstrap for time series data.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    block_length : int
        Length of blocks to resample
    num_bootstrap : int
        Number of bootstrap samples
    
    Returns:
    --------
    np.ndarray
        Bootstrap estimates of regression coefficients
    """
    T = len(y)
    num_blocks = T // block_length + (1 if T % block_length else 0)
    
    # Fit the original model
    X = np.column_stack([np.ones(T), x])
    original_beta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    bootstrap_estimates = np.zeros((num_bootstrap, 2))
    
    for i in range(num_bootstrap):
        # Create bootstrap sample
        bootstrap_indices = np.random.choice(
            np.arange(T - block_length + 1), 
            size=num_blocks, 
            replace=True
        )
        bootstrap_sample_indices = np.hstack([
            np.arange(index, index + block_length) 
            for index in bootstrap_indices
        ])
        bootstrap_sample_indices = bootstrap_sample_indices[:T]
        
        x_bootstrap = x[bootstrap_sample_indices]
        y_bootstrap = y[bootstrap_sample_indices]
        
        # Refit the model on bootstrap sample
        X_bootstrap = np.column_stack([np.ones(T), x_bootstrap])
        bootstrap_beta = np.linalg.inv(X_bootstrap.T @ X_bootstrap) @ X_bootstrap.T @ y_bootstrap
        bootstrap_estimates[i, :] = bootstrap_beta
    
    return bootstrap_estimates

def monte_carlo_simulation(T: int, num_simulations: int, num_bootstrap: int, 
                         block_length: int, params: Dict) -> Dict:
    """
    Perform Monte Carlo simulation with bootstrap inference.
    
    Parameters:
    -----------
    T : int
        Sample size
    num_simulations : int
        Number of Monte Carlo simulations
    num_bootstrap : int
        Number of bootstrap samples per simulation
    block_length : int
        Length of blocks for bootstrap
    params : Dict
        Dictionary containing simulation parameters
    
    Returns:
    --------
    Dict
        Results of the Monte Carlo simulation
    """
    bootstrap_coverage = []
    theoretical_coverage = []
    
    for i in range(num_simulations):
        # Simulate data
        x, y, u = simulate_regression_with_ar1_errors(
            T, params['beta0'], params['beta1'],
            params['phi_x'], params['phi_u'], params['sigma']
        )
        
        # Fit original model
        X = np.column_stack([np.ones(T), x])
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        u_hat = y - X @ beta_hat
        sigma2_hat = np.sum(u_hat**2) / (T - 2)
        var_beta = sigma2_hat * np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(var_beta[1, 1])
        
        # Theoretical confidence interval
        theoretical_ci = (
            beta_hat[1] - 1.96 * se_beta1,
            beta_hat[1] + 1.96 * se_beta1
        )
        theoretical_coverage.append(
            theoretical_ci[0] < params['beta1'] < theoretical_ci[1]
        )
        
        # Bootstrap confidence interval
        bootstrap_estimates = moving_block_bootstrap(x, y, block_length, num_bootstrap)
        bootstrap_ci = np.percentile(bootstrap_estimates[:, 1], [2.5, 97.5])
        bootstrap_coverage.append(
            bootstrap_ci[0] < params['beta1'] < bootstrap_ci[1]
        )
    
    return {
        'bootstrap_coverage': np.mean(bootstrap_coverage),
        'theoretical_coverage': np.mean(theoretical_coverage),
        'bootstrap_estimates': bootstrap_estimates
    }

def plot_results(results: Dict, T: int, save_path: str):
    """
    Plot the results of the Monte Carlo simulation.
    
    Parameters:
    -----------
    results : Dict
        Results from the Monte Carlo simulation
    T : int
        Sample size
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot bootstrap distribution
    plt.subplot(1, 2, 1)
    sns.histplot(results['bootstrap_estimates'][:, 1], kde=True)
    plt.axvline(x=2.0, color='r', linestyle='--', label='True β₁')
    plt.title(f'Bootstrap Distribution of β₁ (T={T})')
    plt.xlabel('β₁')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot coverage rates
    plt.subplot(1, 2, 2)
    coverage_data = pd.DataFrame({
        'Method': ['Bootstrap', 'Theoretical'],
        'Coverage': [results['bootstrap_coverage'], results['theoretical_coverage']]
    })
    sns.barplot(data=coverage_data, x='Method', y='Coverage')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
    plt.title(f'Coverage Rates (T={T})')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Simulation parameters
    params = {
        'beta0': 1.0,
        'beta1': 2.0,
        'phi_x': 0.7,
        'phi_u': 0.7,
        'sigma': 1.0
    }
    
    # Run simulations for different sample sizes
    sample_sizes = [100, 500]
    num_simulations = 1000
    num_bootstrap = 1000
    block_length = 12
    
    for T in sample_sizes:
        print(f"\nRunning simulation for T={T}")
        results = monte_carlo_simulation(
            T, num_simulations, num_bootstrap, block_length, params
        )
        
        print(f"Bootstrap coverage rate: {results['bootstrap_coverage']:.3f}")
        print(f"Theoretical coverage rate: {results['theoretical_coverage']:.3f}")
        
        # Plot results
        plot_results(
            results, 
            T, 
            f'outputs/monte_carlo_results_T{T}.png'
        )

if __name__ == "__main__":
    main() 