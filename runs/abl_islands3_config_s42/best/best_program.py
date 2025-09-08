# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A hybrid search algorithm combining global exploration and local exploitation.
    
    Performs both random global search and directed local search with adaptive step size
    to efficiently find function minima while avoiding local optima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value) - Coordinates and value of the minimum found
    """
    # Initialize with a random point using direct assignment
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Simple random search
        # Generate candidate point using direct assignment
        x, y = np.random.uniform(bounds[0], bounds[1], size=2)
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y
            
        # Local search around best solution with decreasing step size
        step_factor = 0.05 * (1 - _/iterations)  # Start at 0.05 and decrease to 0
        step_size = (bounds[1] - bounds[0]) * step_factor
        x_local = best_x + np.random.uniform(-step_size, step_size)
        y_local = best_y + np.random.uniform(-step_size, step_size)
        
        # Keep within bounds
        x_local = np.clip(x_local, bounds[0], bounds[1])
        y_local = np.clip(y_local, bounds[0], bounds[1])
        
        local_value = evaluate_function(x_local, y_local)
        if local_value < best_value:
            best_value = local_value
            best_x, best_y = x_local, y_local

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
