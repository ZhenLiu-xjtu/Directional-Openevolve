# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Random search algorithm to find function minimum within specified bounds.
    
    Randomly samples points in the search space and tracks the minimum value found.
    
    Args:
        iterations: Number of search iterations
        bounds: Tuple containing (min, max) search bounds
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    lower, upper = bounds
    best_x, best_y = np.random.uniform(lower, upper, size=2)
    best_value = evaluate_function(best_x, best_y)

    for i in range(iterations):
        # Primary exploration: global random search (90% of iterations)
        if i % 10 != 0:
            x, y = np.random.uniform(lower, upper, size=2)
        # Secondary exploration: local search around best solution (10% of iterations)
        else:
            # Add small Gaussian perturbation to best solution
            x = np.clip(best_x + np.random.normal(0, 0.3), lower, upper)
            y = np.clip(best_y + np.random.normal(0, 0.3), lower, upper)
            
        value = evaluate_function(x, y)

        # Update best solution if current is better
        if value < best_value:
            best_value, best_x, best_y = value, x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")
