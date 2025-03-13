# Steps Taken for Assignment 2: Optimization Techniques

## 1. Project Setup and Planning

[+] Created a new Python file `assignment2_python.py` for implementing optimization techniques.

[+] Imported necessary libraries:
- NumPy for numerical computations
- Pandas for data handling
- Matplotlib for visualization
- SciPy for optimization utilities

[+] Created an outputs directory to store visualization results.

## 2. Steepest Descent Algorithm Implementation

[+] Implemented the steepest descent optimization algorithm with the following features:
  - Function parameters:
    - Objective function to minimize
    - Gradient function
    - Initial starting point
    - Learning rate (step size)
    - Maximum number of iterations
    - Convergence criterion

  - Return values:
    - Optimized solution
    - History of iterations (x, f(x), ||∇f(x)||)

  - Algorithm steps:
    1. Initialize with starting point
    2. For each iteration:
       - Calculate gradient at current point
       - Update position using gradient and learning rate
       - Calculate new function value and gradient norm
       - Record iteration history
       - Check convergence criterion
    3. Return optimized solution and history

## 3. Test Function Definition

[+] Defined a simple test objective function: f(x) = Σ(x_i - 3)^2
  - This function has a known global minimum at x_i = 3 for all dimensions
  - Function is smooth and convex, making it suitable for gradient-based optimization

[+] Implemented the analytical gradient: ∇f(x) = 2(x - 3)
  - Exact gradient calculation for performance comparison

## 4. Gradient Approximation Implementation

[+] Implemented finite difference approximation for gradient calculation:
  - Used SciPy's approx_fprime utility
  - Set appropriate epsilon value (sqrt of machine precision)
  - This allows comparison between analytical and numerical approaches

## 5. Comparison Framework Development

[+] Created a function to compare different gradient calculation methods:
  - Measures computation time for both methods
  - Calculates error between finite difference and analytical gradients
  - Tests across different problem dimensions to analyze scaling behavior

[+] Designed comparison to capture:
  - Accuracy trade-offs
  - Computational efficiency
  - Scaling with problem dimension

## 6. Main Execution Setup

[+] Configured two example demonstrations:

  - Example 1: Steepest Descent Optimization
    - Set initial guess at [0.0, 0.0]
    - Used learning rate of 0.2
    - Ran optimization until convergence
    - Printed final solution and objective value

  - Example 2: Gradient Method Comparison
    - Tested dimensions from 1 to 50 (in steps of 5)
    - Calculated errors and timing for both methods
    - Stored results for visualization

## 7. Visualization Implementation

[+] Created comprehensive visualizations to illustrate optimization performance:

  - Convergence history plots:
    - Objective function value vs. iteration (semilogy scale)
    - Gradient norm vs. iteration (semilogy scale)
    - These show how quickly the algorithm approaches the solution

  - Gradient method comparison plots:
    - Finite difference error vs. dimension
    - Computation time vs. dimension for both methods
    - These illustrate the trade-offs between methods

[+] Applied proper formatting:
  - Titles, labels, and legends
  - Grid lines for readability
  - Appropriate scales (logarithmic where needed)
  - Multiple subplots for clear comparison

## 8. Testing and Verification

[+] Verified the steepest descent implementation:
  - Confirmed convergence to known solution [3.0, 3.0]
  - Checked that gradient norm approaches zero
  - Ensured objective function value minimizes appropriately

[+] Validated gradient comparison:
  - Confirmed reasonable error values for finite difference
  - Verified computational scaling behavior with dimension

## 9. Output Management

[+] Set up automatic saving of visualization results:
  - Created an outputs directory if it doesn't exist
  - Saved plots as PNG files with descriptive names
  - Ensured plots are properly closed after saving

[+] Added informative console output:
  - Iteration progress during optimization
  - Final solution and objective value
  - Confirmation of saved results

## 10. Final Review and Organization

[+] Conducted code review:
  - Added comprehensive docstrings and comments
  - Ensured consistent formatting and naming conventions
  - Verified all code sections work together correctly

[+] Prepared documentation including this STEPS.md file to explain the implementation process and decisions made throughout the project. 