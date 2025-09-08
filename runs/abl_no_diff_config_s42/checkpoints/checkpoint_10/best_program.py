# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simplified hybrid search algorithm combining random exploration and local refinement.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point using array for cleaner coordinates handling
    best_point = np.random.uniform(bounds[0], bounds[1], 2)
    best_value = evaluate_function(*best_point)
    
    for _ in range(iterations):
        # Hybrid search: random exploration (70%) or local refinement (30%)
        if np.random.rand() < 0.7:
            # Global exploration with uniform random sampling
            new_point = np.random.uniform(bounds[0], bounds[1], 2)
        else:
            # Local refinement around current best with adaptive perturbation
            perturbation = np.random.normal(0, (bounds[1]-bounds[0])/20, 2)
            new_point = np.clip(best_point + perturbation, bounds[0], bounds[1])
        
        # Evaluate new point and update if better
        current_value = evaluate_function(*new_point)
        if current_value < best_value:
            best_point, best_value = new_point, current_value
    
    return (*best_point, best_value)


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