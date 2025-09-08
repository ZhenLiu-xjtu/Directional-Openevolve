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
    num_initial_points = 5
    initial_points = np.random.uniform(bounds[0], bounds[1], size=(num_initial_points, 2))
    initial_values = [evaluate_function(x, y) for x, y in initial_points]
    
    # Select best initial point
    best_idx = np.argmin(initial_values)
    best_x, best_y = initial_points[best_idx]
    best_value = initial_values[best_idx]

    for i in range(iterations):
        # Adaptive exploration-exploitation balance: start with exploration, shift to exploitation
        exploration_rate = max(0.1, 1.0 - i/iterations)  # Decreases from 1.0 to 0.1
        
        # Adaptive search: global exploration with local refinement
        if np.random.random() < exploration_rate:  # Adaptive exploration probability
            # Adaptive Gaussian perturbation around current best with decreasing step size
            step_size = 0.3 * exploration_rate  # Step size decreases as exploration decreases
            x = np.clip(best_x + np.random.normal(0, step_size), bounds[0], bounds[1])
            y = np.clip(best_y + np.random.normal(0, step_size), bounds[0], bounds[1])
        else:  # 80% chance for global exploration
            x, y = np.random.uniform(bounds[0], bounds[1], size=2)
            
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
