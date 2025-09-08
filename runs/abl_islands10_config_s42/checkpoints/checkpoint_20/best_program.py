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
    low, high = bounds
    # Initialize with random starting point
    best_x, best_y = np.random.uniform(low, high, size=2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Generate new candidate point
        candidate_x, candidate_y = np.random.uniform(low, high, size=2)
        candidate_value = evaluate_function(candidate_x, candidate_y)

        if candidate_value < best_value:
            best_value = candidate_value
            best_x, best_y = candidate_x, candidate_y

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
