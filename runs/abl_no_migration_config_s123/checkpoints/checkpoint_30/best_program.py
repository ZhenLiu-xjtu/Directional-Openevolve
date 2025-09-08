# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5), local_search_prob=0.3, local_search_range=0.5):
    """
    An improved random search algorithm with local search to avoid local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        local_search_prob: Probability of searching near best solution (0-1)
        local_search_range: Range for local search exploration

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], 2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # With probability, search locally around best solution
        if np.random.random() < local_search_prob:
            # Generate candidate near best solution with normal distribution
            x = np.clip(best_x + np.random.normal(0, local_search_range), bounds[0], bounds[1])
            y = np.clip(best_y + np.random.normal(0, local_search_range), bounds[0], bounds[1])
        else:
            # Generate random coordinates globally
            x, y = np.random.uniform(bounds[0], bounds[1], 2)
        value = evaluate_function(x, y)

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
