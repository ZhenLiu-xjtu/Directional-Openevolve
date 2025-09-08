# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A hybrid search algorithm combining random exploration with local refinement.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point
    best_x, best_y = np.random.uniform(*bounds, size=2)
    best_value = evaluate_function(best_x, best_y)
    
    # Probability of performing local search around best solution
    local_search_prob = 0.2
    
    for _ in range(iterations):
        # Adaptive search strategy
        if np.random.random() < local_search_prob:
            # Local search: explore around best solution
            search_width = (bounds[1] - bounds[0]) / 10  # 10% of search space
            x = np.clip(np.random.normal(best_x, search_width), *bounds)
            y = np.clip(np.random.normal(best_y, search_width), *bounds)
        else:
            # Global search: explore new random point
            x, y = np.random.uniform(*bounds, size=2)
            
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
