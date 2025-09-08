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
    # Initialize with a random point
    low, high = bounds
    best_x, best_y = np.random.uniform(low, high, size=2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Adaptive search: balance exploration and exploitation
        if _ > 0 and np.random.random() < 0.3:  # 30% local search around best
            # Decreasing step size for convergence
            step_size = 0.1 * (1 - _/iterations)
            x = best_x + np.random.normal(0, step_size)
            y = best_y + np.random.normal(0, step_size)
            x, y = np.clip(x, low, high), np.clip(y, low, high)
        else:  # 70% global exploration
            x, y = np.random.uniform(low, high, size=2)
            
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
