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
    best_x, best_y = np.random.uniform(*bounds, size=2)
    best_value = evaluate_function(best_x, best_y)

    for i in range(iterations):
        # Adaptive search: balance global exploration and local exploitation
        exploration_rate = max(0.1, 1.0 - i/iterations)  # Decreases from 1.0 to 0.1
        
        if np.random.random() < 0.3:  # 30% global exploration
            current_x, current_y = np.random.uniform(*bounds, size=2)
        else:  # 70% local exploration around best solution
            radius = exploration_rate * (bounds[1] - bounds[0])/4
            current_x = np.clip(best_x + np.random.normal(0, radius), bounds[0], bounds[1])
            current_y = np.clip(best_y + np.random.normal(0, radius), bounds[0], bounds[1])
            
        current_value = evaluate_function(current_x, current_y)

        if current_value < best_value:
            best_value = current_value
            best_x, best_y = current_x, current_y

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
