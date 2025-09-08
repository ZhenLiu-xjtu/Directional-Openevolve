# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An optimized search algorithm with adaptive exploration-exploitation balance.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random coordinates
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], 2)
    best_value = evaluate_function(best_x, best_y)
    
    min_bound, max_bound = bounds
    search_range = max_bound - min_bound

    for i in range(iterations):
        # Adaptively balance exploration and exploitation
        # Start with more exploration, gradually shift to exploitation
        exploration_rate = 0.5 * (1 - i/iterations) + 0.1  # 10-60% exploration
        
        if np.random.random() < exploration_rate:
            # Exploration: random sampling across entire search space
            x, y = np.random.uniform(min_bound, max_bound, 2)
        else:
            # Exploitation: local search around best found solution
            search_width = search_range * (0.2 * (1 - i/iterations) + 0.05)
            x = np.clip(np.random.normal(best_x, search_width/3), min_bound, max_bound)
            y = np.clip(np.random.normal(best_y, search_width/3), min_bound, max_bound)
            
        current_value = evaluate_function(x, y)

        # Update best if current point is better
        if current_value < best_value:
            best_x, best_y, best_value = x, y, current_value

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