# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np
from typing import Tuple

def search_algorithm(iterations: int = 1000, bounds: Tuple[float, float] = (-5, 5)) -> Tuple[float, float, float]:
    """
    A search algorithm that balances exploration and exploitation to find function minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point (simplified)
    best_x, best_y = np.random.uniform(*bounds, 2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Balance exploration (global search) and exploitation (local refinement)
        if np.random.random() < 0.7:  # 70% local exploitation
            # Adaptive step size that decreases over time
            step = 0.1 * (1 - _/iterations) * (bounds[1] - bounds[0])
            x, y = np.clip([best_x, best_y] + np.random.normal(0, step, 2), bounds[0], bounds[1])
        else:  # 30% global exploration
            x, y = np.random.uniform(*bounds, 2)
            
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
