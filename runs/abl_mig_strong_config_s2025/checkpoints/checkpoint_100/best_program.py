# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Adaptive random search algorithm for function minimization within specified bounds.
    
    Combines global exploration with local refinement around promising solutions.

    Args:
        iterations: Number of iterations to run (default: 1000)
        bounds: Tuple specifying search space boundaries (min, max) (default: (-5, 5))

    Returns:
        Tuple containing (best_x, best_y, best_value)
    """
    # Unpack bounds and initialize with random point
    lower_bound, upper_bound = bounds
    best_x, best_y = np.random.uniform(lower_bound, upper_bound, size=2)
    best_value = evaluate_function(best_x, best_y)

    # Search parameters
    LOCAL_SEARCH_PROB = 0.2  # Probability of local vs global search
    PERTURBATION_STD = 0.5   # Standard deviation for local search perturbations
    
    for _ in range(iterations):
        # Adaptive search: explore globally and refine locally
        if np.random.random() < LOCAL_SEARCH_PROB:  # Local refinement
            # Add small perturbation to best solution
            x = best_x + np.random.normal(0, PERTURBATION_STD)
            y = best_y + np.random.normal(0, PERTURBATION_STD)
            # Keep within bounds
            x = np.clip(x, lower_bound, upper_bound)
            y = np.clip(y, lower_bound, upper_bound)
        else:  # Global exploration
            x, y = np.random.uniform(lower_bound, upper_bound, size=2)
            
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
