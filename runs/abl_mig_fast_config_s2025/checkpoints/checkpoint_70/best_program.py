# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Simplified adaptive search algorithm that finds function minima by balancing exploration and exploitation.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(best_x, best_y)
    
    # Adaptive parameters
    initial_radius = (bounds[1] - bounds[0]) / 2
    final_radius = initial_radius * 0.01

    for i in range(iterations):
        # Adaptive radius - starts wide, gets narrower
        radius = initial_radius - (initial_radius - final_radius) * (i / iterations)
        
        # Search around best point with current radius
        x = np.clip(best_x + np.random.normal(0, radius), bounds[0], bounds[1])
        y = np.clip(best_y + np.random.normal(0, radius), bounds[0], bounds[1])
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
