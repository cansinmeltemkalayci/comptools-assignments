#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 2 Python Implementation - Optimization Techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.optimize import approx_fprime
import os

# Create outputs directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Define the steepest descent algorithm
def steepest_descent(f, gradient, initial_guess, learning_rate, num_iterations=100, epsilon_g=1e-7):
    """
    Steepest descent optimization algorithm.
    
    Parameters:
    -----------
    f : function
        Objective function to minimize
    gradient : function
        Function to compute gradient of f
    initial_guess : array_like
        Starting point for the optimization
    learning_rate : float
        Step size for the gradient descent
    num_iterations : int, optional
        Maximum number of iterations
    epsilon_g : float, optional
        Convergence criterion for gradient norm
        
    Returns:
    --------
    x : array_like
        Approximate solution
    history : list
        History of iterations including x, f(x), and ||g(x)||
    """
    x = np.array(initial_guess, dtype=float)
    history = []
    
    for i in range(num_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        norm_g = np.linalg.norm(grad)
        f_value = f(x)
        
        history.append({
            'iteration': i+1,
            'x': x.copy(),
            'f': f_value,
            'grad_norm': norm_g
        })
        
        print(f"Iteration {i+1}: x = {x}, f(x) = {f_value}, ||g(x)||={norm_g}")
        
        # Termination condition
        if norm_g < epsilon_g:
            break
            
    return x, history

# Define test function f(x) = Σ(x_i - 3)^2
def objective_function(x):
    """
    Objective function: sum of squared distances from 3
    """
    return np.sum((x - 3.0)**2)

# Analytical gradient of the objective function
def analytical_gradient(x):
    """
    Analytical gradient of the objective function
    """
    return 2 * (x - 3.0)

# Finite difference approximation to gradient
def finite_difference_gradient(f, x, epsilon=np.sqrt(np.finfo(float).eps)):
    """
    Compute the gradient of function f at point x using finite differences
    """
    return approx_fprime(x, f, epsilon)

# Function to calculate computation time and error for different gradient methods
def compare_gradient_methods(f, analytical_grad, x, dimensions_range):
    """
    Compare finite difference and analytical gradient methods
    
    Parameters:
    -----------
    f : function
        Objective function
    analytical_grad : function
        Analytical gradient function
    x : array_like
        Point at which to evaluate the gradient
    dimensions_range : range
        Range of dimensions to test
        
    Returns:
    --------
    dict
        Results of comparison including errors and times
    """
    results = {
        'dimensions': list(dimensions_range),
        'fd_errors': [],
        'fd_times': [],
        'analytical_times': []
    }
    
    for dim in dimensions_range:
        x_dim = np.random.randn(dim)
        
        # Analytical gradient (ground truth)
        start_time = time.time()
        true_grad = analytical_grad(x_dim)
        analytical_time = time.time() - start_time
        
        # Finite difference gradient
        start_time = time.time()
        fd_grad = finite_difference_gradient(f, x_dim)
        fd_time = time.time() - start_time
        
        # Error calculation
        error = np.linalg.norm(fd_grad - true_grad)
        
        # Store results
        results['fd_errors'].append(error)
        results['fd_times'].append(fd_time)
        results['analytical_times'].append(analytical_time)
    
    return results

# Main execution
if __name__ == "__main__":
    print("Assignment 2 - Optimization Techniques")
    
    # Example 1: Optimize the test function using steepest descent
    initial_guess = np.array([0.0, 0.0])
    learning_rate = 0.2
    
    print("\nExample 1: Steepest Descent Optimization")
    solution, history = steepest_descent(
        objective_function, 
        analytical_gradient, 
        initial_guess, 
        learning_rate
    )
    
    print(f"\nFinal solution: {solution}")
    print(f"Final objective value: {objective_function(solution)}")
    
    # Plot the convergence history
    iterations = [h['iteration'] for h in history]
    f_values = [h['f'] for h in history]
    grad_norms = [h['grad_norm'] for h in history]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(iterations, f_values)
    plt.title('Objective Function Value')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(iterations, grad_norms)
    plt.title('Gradient Norm')
    plt.xlabel('Iteration')
    plt.ylabel('||∇f(x)||')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/convergence_history.png')
    plt.close()
    
    # Example 2: Compare gradient calculation methods
    print("\nExample 2: Comparing Gradient Calculation Methods")
    dimensions = range(1, 51, 5)  # Test with dimensions 1, 6, 11, ..., 46
    comparison_results = compare_gradient_methods(
        objective_function,
        analytical_gradient,
        np.array([1.0, 2.0]),
        dimensions
    )
    
    # Plot the comparison results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(comparison_results['dimensions'], comparison_results['fd_errors'])
    plt.title('Finite Difference Error')
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(comparison_results['dimensions'], comparison_results['fd_times'], label='Finite Difference')
    plt.plot(comparison_results['dimensions'], comparison_results['analytical_times'], label='Analytical')
    plt.title('Computation Time')
    plt.xlabel('Dimension')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/gradient_comparison.png')
    plt.close()
    
    print("\nResults saved to outputs directory.") 