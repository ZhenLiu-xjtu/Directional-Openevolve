# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1500, bounds=(-5, 5)):
    """
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    min_bound, max_bound = bounds
    best_x, best_y = np.random.uniform(min_bound, max_bound, size=2)
    best_value = evaluate_function(best_x, best_y)
    
    for _ in range(iterations):
        # Adaptive search parameters
        exploration_ratio = 0.3  # 30% local, 70% global search
        current_std = 0.5 * (1 - _/iterations)  # Decreasing perturbation
        
        if np.random.random() < exploration_ratio:
            # Local exploration with adaptive perturbation
            x = np.clip(best_x + np.random.normal(0, current_std), min_bound, max_bound)
            y = np.clip(best_y + np.random.normal(0, current_std), min_bound, max_bound)
        else:
            # Global exploration with random sampling
            x, y = np.random.uniform(min_bound, max_bound, size=2)
            
        value = evaluate_function(x, y)

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
