# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np
from typing import Tuple


def search_algorithm(iterations: int = 1000, bounds: Tuple[float, float] = (-5, 5)) -> Tuple[float, float, float]:
    """
    A simple random search algorithm to find the minimum of a function.

    Args:
        iterations: Number of iterations to run (default: 1000)
        bounds: Bounds for the search space as (min, max) (default: (-5, 5))

    Returns:
        Tuple containing the best x, best y, and the corresponding minimum value
    """
    # Initialize with multiple random points to find better starting position
    best_value = float('inf')
    best_x, best_y = None, None
    
    # Try 5 initial points
    for _ in range(5):
        x, y = np.random.uniform(bounds[0], bounds[1], 2)
        value = evaluate_function(x, y)
        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    for _ in range(iterations):
        # Adaptive search: explore around current best (20% chance) or random search
        if np.random.random() < 0.2:
            # Local search with decreasing step size
            step_size = (bounds[1] - bounds[0]) * (1 - _/iterations) * 0.1
            x = best_x + np.random.normal(0, step_size)
            y = best_y + np.random.normal(0, step_size)
            x, y = np.clip(x, bounds[0], bounds[1]), np.clip(y, bounds[0], bounds[1])
        else:
            # Global random search
            x, y = np.random.uniform(bounds[0], bounds[1], 2)
            
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
