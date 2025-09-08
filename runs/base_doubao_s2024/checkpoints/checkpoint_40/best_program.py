# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Balanced search algorithm with reduced local exploration for better global coverage.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random 2D point
    best_point = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(*best_point)

    for _ in range(iterations):
        # Generate random point or local perturbation of best point (10% chance)
        if np.random.rand() < 0.05:  # Reduced to 5% for better global exploration
            point = best_point + np.random.normal(0, 0.2, size=2)  # Smaller perturbation
            point = np.clip(point, bounds[0], bounds[1])  # Keep within bounds
        else:
            point = np.random.uniform(bounds[0], bounds[1], size=2)
            
        value = evaluate_function(*point)

        if value < best_value:
            best_value = value
            best_point = point.copy()

    return best_point[0], best_point[1], best_value


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
