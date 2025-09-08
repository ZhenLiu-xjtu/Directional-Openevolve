# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Hybrid random search optimization algorithm for function minimization.
    
    Explores the search space through a combination of global random search
    and local search around promising regions, gradually focusing on the most
    promising areas over time.

    Args:
        iterations: Number of search iterations (default: 1000)
        bounds: Tuple specifying search space boundaries as (min, max) (default: (-5, 5))

    Returns:
        Tuple containing (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(best_x, best_y)

    for i in range(iterations):
        # Hybrid search: mostly global, occasionally local around best point
        if np.random.random() < 0.1:  # 10% chance of local search
            # Local search with adaptive radius that shrinks over time
            radius = 0.5 * (1 - i/iterations)  # Shrinking radius from 0.5 to 0
            x = np.clip(best_x + np.random.normal(0, radius), bounds[0], bounds[1])
            y = np.clip(best_y + np.random.normal(0, radius), bounds[0], bounds[1])
        else:
            # Global random search
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
