# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize bounds and starting point
    min_bound, max_bound = bounds
    best_x, best_y = np.random.uniform(min_bound, max_bound, 2)
    best_value = evaluate_function(best_x, best_y)

    for i in range(iterations):
        # Adaptive search: More exploration early, more exploitation later
        exploration_ratio = 0.3 * (1 - i/iterations)  # Decreases from 0.3 to 0
        if np.random.random() < exploration_ratio:
            # Narrow search over time: starts broad, becomes focused
            search_width = (max_bound - min_bound) / 10 * (1 - i/iterations)
            x = best_x + np.random.normal(0, search_width)
            y = best_y + np.random.normal(0, search_width)
            x, y = np.clip(x, min_bound, max_bound), np.clip(y, min_bound, max_bound)
        else:
            # Random search in bounds
            x, y = np.random.uniform(min_bound, max_bound, 2)
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
