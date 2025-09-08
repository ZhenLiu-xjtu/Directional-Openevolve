# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An optimized random search algorithm with local exploration.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point using concise syntax
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], 2)
    best_value = evaluate_function(best_x, best_y)
    
    # Search space boundaries
    min_bound, max_bound = bounds
    
    for _ in range(iterations):
        # Adaptive search strategy: sometimes explore locally around best found point
        if np.random.rand() < 0.2:  # 20% chance of local exploration
            # Perturb around best point with decreasing step size
            step = 0.1 * (1 - _/iterations) * (max_bound - min_bound)
            x = np.clip(best_x + np.random.normal(0, step), min_bound, max_bound)
            y = np.clip(best_y + np.random.normal(0, step), min_bound, max_bound)
        else:  # 80% chance of global exploration
            x, y = np.random.uniform(min_bound, max_bound, 2)
            
        value = evaluate_function(x, y)
        
        # Update best solution if improved
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
    
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