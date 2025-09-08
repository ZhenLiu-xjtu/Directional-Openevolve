# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An optimized search algorithm with balanced exploration and exploitation.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point and evaluate
    best_x, best_y = np.random.uniform(*bounds, 2)
    best_value = evaluate_function(best_x, best_y)
    min_bound, max_bound = bounds
    bound_range = max_bound - min_bound
    
    for i in range(iterations):
        # Generate candidate solution using adaptive search strategy
        rand_val = np.random.rand()
        
        if rand_val < 0.3:  # Local exploration (30%)
            # Perturb around best point with adaptive step size
            step = 0.12 * (1 - i/iterations) * bound_range
            x = np.clip(best_x + np.random.normal(0, step), min_bound, max_bound)
            y = np.clip(best_y + np.random.normal(0, step), min_bound, max_bound)
        elif rand_val < 0.5:  # Directed exploration from best point (20%)
            x, y = np.random.normal([best_x, best_y], bound_range * 0.1)
            x, y = np.clip(x, min_bound, max_bound), np.clip(y, min_bound, max_bound)
        else:  # Pure random exploration (50%)
            x, y = np.random.uniform(min_bound, max_bound, 2)
            
        # Evaluate candidate solution and update best if improved
        current_value = evaluate_function(x, y)
        if current_value < best_value:
            best_value, best_x, best_y = current_value, x, y
    
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