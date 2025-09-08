# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Optimized search algorithm combining global exploration and local exploitation.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point and extract bounds
    min_bound, max_bound = bounds
    best_x, best_y = np.random.uniform(min_bound, max_bound, 2)
    best_value = evaluate_function(best_x, best_y)
    
    for i in range(iterations):
        # Adaptive search strategy with 20% local/80% global exploration
        if np.random.rand() < 0.2:
            # Local exploration with decreasing step size
            step = 0.1 * (1 - i/iterations) * (max_bound - min_bound)
            x = np.clip(best_x + np.random.normal(0, step), min_bound, max_bound)
            y = np.clip(best_y + np.random.normal(0, step), min_bound, max_bound)
        else:
            # Global exploration
            x, y = np.random.uniform(min_bound, max_bound, 2)
            
        # Update best solution if improved
        current_value = evaluate_function(x, y)
        if current_value < best_value:
            best_x, best_y, best_value = x, y, current_value
    
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