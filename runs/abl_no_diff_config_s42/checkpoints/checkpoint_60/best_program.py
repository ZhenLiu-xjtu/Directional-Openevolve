# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simplified yet effective search algorithm balancing exploration and exploitation.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point and evaluate
    best_point = np.random.uniform(bounds[0], bounds[1], 2)
    best_value = evaluate_function(*best_point)
    search_range = bounds[1] - bounds[0]

    for i in range(iterations):
        # Adaptive exploration-exploitation balance (more exploitation over time)
        explore_prob = 0.8 - 0.5 * (i/iterations)
        
        if np.random.random() < explore_prob:
            # Global exploration with random sampling
            point = np.random.uniform(bounds[0], bounds[1], 2)
        else:
            # Local exploitation with adaptive step size (decreases over time)
            step_size = search_range * (0.05 - 0.03 * (i/iterations))
            point = best_point + np.random.normal(0, step_size, 2)
            point = np.clip(point, bounds[0], bounds[1])
        
        # Evaluate and update if improved
        current_value = evaluate_function(*point)
        if current_value < best_value:
            best_point, best_value = point, current_value

    return (*best_point, best_value)


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