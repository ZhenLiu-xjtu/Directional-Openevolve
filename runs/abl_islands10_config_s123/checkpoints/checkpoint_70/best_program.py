# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A random search algorithm with local exploration for function minimization.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Unpack bounds for readability
    lower_bound, upper_bound = bounds
    
    # Initialize with a random starting point
    best_x, best_y = np.random.uniform(lower_bound, upper_bound, size=2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Adaptive search: sometimes explore locally around best solution
        if np.random.rand() < 0.2:  # 20% chance of local exploration
            # Local exploration with decreasing step size
            step_size = 0.1 * (1 - _/iterations) * (upper_bound - lower_bound)
            x = best_x + np.random.normal(0, step_size)
            y = best_y + np.random.normal(0, step_size)
            # Ensure we stay within bounds
            x = np.clip(x, lower_bound, upper_bound)
            y = np.clip(y, lower_bound, upper_bound)
        else:
            # Global exploration
            x, y = np.random.uniform(lower_bound, upper_bound, size=2)
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
