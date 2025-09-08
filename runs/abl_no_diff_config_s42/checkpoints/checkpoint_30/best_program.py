# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An optimized search algorithm with global exploration and local refinement.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random 2D point
    best_point = np.random.uniform(bounds[0], bounds[1], size=2)
    best_value = evaluate_function(*best_point)
    
    min_bound, max_bound = bounds
    search_range = max_bound - min_bound

    for _ in range(iterations):
        # With 20% probability, explore locally around best point
        if np.random.random() < 0.2:
            # Local exploration with adaptive step size (5% of search range)
            step_size = search_range * 0.05
            point = best_point + np.random.normal(0, step_size, size=2)
            point = np.clip(point, min_bound, max_bound)  # Keep within bounds
        else:
            # Global exploration with random point
            point = np.random.uniform(min_bound, max_bound, size=2)
            
        # Evaluate candidate point
        value = evaluate_function(*point)

        # Update best solution if improved
        if value < best_value:
            best_point = point
            best_value = value

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
    print(f"Found minimum at ({x}, {y}) with value {value}")