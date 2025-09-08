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
    best_x, best_y = np.random.uniform(*bounds, size=2)
    best_value = evaluate_function(best_x, best_y)
    no_improvement_count = 0
    
    # Adaptive search with exploration/exploitation balance
    for i in range(iterations):
        exploration_ratio = 1.0 - (i / iterations)
        
        # Periodic random restart to escape local minima
        if no_improvement_count > iterations // 5:
            x, y = np.random.uniform(*bounds, size=2)
            no_improvement_count = 0
        elif np.random.random() < exploration_ratio:
            # Exploration: random search across entire bounds
            x, y = np.random.uniform(*bounds, size=2)
        else:
            # Exploitation: local search around best found point
            search_width = (bounds[1] - bounds[0]) * exploration_ratio ** 2 / 2
            x = np.clip(best_x + np.random.normal(0, search_width), *bounds)
            y = np.clip(best_y + np.random.normal(0, search_width), *bounds)
            
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y
            no_improvement_count = 0
        else:
            no_improvement_count += 1

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
