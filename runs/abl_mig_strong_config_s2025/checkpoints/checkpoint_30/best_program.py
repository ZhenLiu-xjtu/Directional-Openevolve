# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Random search algorithm for function minimization within specified bounds.

    Args:
        iterations: Number of iterations to run (default: 1000)
        bounds: Tuple specifying search space boundaries (min, max) (default: (-5, 5))

    Returns:
        Tuple containing (best_x, best_y, best_value)
    """
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Simple random search (simplified)
        x, y = np.random.uniform(bounds[0], bounds[1], size=2)
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
