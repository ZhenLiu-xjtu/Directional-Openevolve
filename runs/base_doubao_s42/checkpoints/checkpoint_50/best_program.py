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
    # Initialize with random point
    best_coords = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(*best_coords)

    for _ in range(iterations):
        # Adaptive search: balance global exploration and local exploitation
        if np.random.rand() < 0.2:  # 20% chance of local exploration
            # Explore locally around best solution with decreasing step size
            step_size = (bounds[1] - bounds[0]) * (1 - _/iterations) * 0.1
            coords = best_coords + np.random.normal(0, step_size, size=2)
            # Keep within bounds
            coords = np.clip(coords, bounds[0], bounds[1])
        else:  # 80% chance of global random search
            coords = np.random.uniform(bounds[0], bounds[1], size=2)
        
        value = evaluate_function(*coords)
        if value < best_value:
            best_value = value
            best_coords = coords

    return best_coords[0], best_coords[1], best_value


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
