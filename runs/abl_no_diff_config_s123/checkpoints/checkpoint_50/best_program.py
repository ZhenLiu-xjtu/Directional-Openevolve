# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An improved search algorithm that balances exploration and exploitation.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point
    best_point = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(*best_point)

    for _ in range(iterations):
        # Balance exploration (random search) and exploitation (local search)
        if np.random.random() < 0.3:  # 30% exploration
            x, y = np.random.uniform(bounds[0], bounds[1], size=2)
        else:  # 70% exploitation - search near current best
            step = np.random.normal(0, 0.5, size=2)
            x, y = np.clip(best_point + step, bounds[0], bounds[1])

        current_value = evaluate_function(x, y)
        if current_value < best_value:
            best_value = current_value
            best_point = np.array([x, y])

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
    print(f"Found minimum at ({x:.4f}, {y:.4f}) with value {value:.4f}")