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
    # Initialize with multiple random points to find better starting position
    best_value = float('inf')
    best_x, best_y = 0, 0
    
    # Initial exploration with multiple points (5% of iterations budget)
    for _ in range(max(10, int(iterations * 0.05))):
        x, y = np.random.uniform(bounds[0], bounds[1], 2)
        value = evaluate_function(x, y)
        
        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    for i in range(iterations):
        # Adaptive search: balance global exploration and local refinement
        if i % 10 == 0:  # Occasional global exploration
            x, y = np.random.uniform(bounds[0], bounds[1], 2)
        else:  # Local exploration around best solution with decreasing step size
            step_size = (bounds[1] - bounds[0]) * (1 - i/iterations)  # Decreasing step size
            x = np.clip(best_x + np.random.normal(0, step_size/3), bounds[0], bounds[1])
            y = np.clip(best_y + np.random.normal(0, step_size/3), bounds[0], bounds[1])
            
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
