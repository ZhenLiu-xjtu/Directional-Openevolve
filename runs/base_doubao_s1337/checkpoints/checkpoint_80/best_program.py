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
    # Initialize with random point and evaluate
    best_x, best_y = np.random.uniform(*bounds, size=2)
    best_value = evaluate_function(best_x, best_y)

    for i in range(iterations):
        # Balance exploration (70%) and exploitation (30%) with adaptive step size
        if np.random.random() < 0.3:  # Local exploitation
            # Adaptive step size decreases over time
            step_size = (bounds[1] - bounds[0]) * (1 - i/iterations) * 0.1
            x = np.clip(best_x + np.random.normal(0, step_size), bounds[0], bounds[1])
            y = np.clip(best_y + np.random.normal(0, step_size), bounds[0], bounds[1])
        else:  # Global exploration
            x, y = np.random.uniform(*bounds, size=2)
            
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
