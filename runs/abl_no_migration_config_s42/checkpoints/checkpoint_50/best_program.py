# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An adaptive search algorithm that balances exploration and exploitation to find function minima.
    
    This algorithm combines local search around promising solutions with global exploration
    to efficiently find minima while avoiding local optima traps.

    Args:
        iterations: Number of iterations to run (default: 1000)
        bounds: Bounds for the search space as (min, max) (default: (-5, 5))

    Returns:
        Tuple containing (best_x, best_y, best_value) representing the found minimum
    """
    # Initialize with a random point
    # Generate initial point with combined random sampling
    best_x, best_y = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(best_x, best_y)

    for iteration in range(iterations):
        # Adaptive search: balance exploration and exploitation
        if iteration > 0 and np.random.random() < 0.3:  # 30% local search around best
            # Decreasing step size for convergence
            step_size = 0.1 * (1 - iteration/iterations)
            # Generate and clip values in one operation
            x, y = np.clip(
                [best_x + np.random.normal(0, step_size), 
                 best_y + np.random.normal(0, step_size)],
                bounds[0], bounds[1]
            )
        else:  # 70% global exploration
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
