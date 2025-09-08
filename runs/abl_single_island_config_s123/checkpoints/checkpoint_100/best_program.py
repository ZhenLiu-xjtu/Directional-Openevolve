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
    best_xy = np.random.uniform(bounds[0], bounds[1], size=2)
    best_x, best_y = best_xy
    best_value = evaluate_function(*best_xy)

    # Add adaptive search parameters
    initial_local_range = (bounds[1] - bounds[0]) * 0.1
    local_search_range = initial_local_range

    for i in range(iterations):
        # Adaptive exploration-exploitation balance (30% to 10%)
        exploration_prob = max(0.1, 0.3 - (i/iterations)*0.2)
        
        if np.random.rand() < exploration_prob:
            # Adaptive local search with decreasing range
            local_search_range = initial_local_range * (1 - i/iterations)
            # Local search using normal distribution for refinement
            xy = best_xy + np.random.normal(0, local_search_range/2, size=2)
            xy = np.clip(xy, bounds[0], bounds[1])
        else:
            # Global random search
            xy = np.random.uniform(bounds[0], bounds[1], size=2)
            
        value = evaluate_function(*xy)
        x, y = xy

        if value < best_value:
            best_value = value
            best_xy = xy
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
