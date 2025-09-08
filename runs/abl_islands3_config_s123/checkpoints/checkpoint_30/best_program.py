# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A hybrid search algorithm with guided local exploration and random search.
    
    30% of iterations explore locally around the best solution found using a 
    normal distribution, while 70% perform random search within the bounds.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point (simplified)
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], 2)
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Guided search: 30% local exploration, 70% random search
        if np.random.random() < 0.3:
            # Explore around best solution with normal distribution
            search_width = (bounds[1] - bounds[0]) / 10  # 10% of search space
            x = best_x + np.random.normal(0, search_width)
            y = best_y + np.random.normal(0, search_width)
            x, y = np.clip(x, bounds[0], bounds[1]), np.clip(y, bounds[0], bounds[1])
        else:
            # Random search in bounds
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
