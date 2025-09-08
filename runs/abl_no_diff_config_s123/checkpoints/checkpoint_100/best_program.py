# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    An improved search algorithm balancing exploration and exploitation.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with random point and evaluate
    min_bound, max_bound = bounds
    best_point = np.random.uniform(min_bound, max_bound, size=2)
    best_value = evaluate_function(*best_point)
    
    # Track recent performance for adaptive adjustments
    recent_improvements = 0
    
    for i in range(iterations):
        # Adapt exploration rate based on recent improvements
        exploration_rate = 0.3 if recent_improvements > 5 else 0.5
        
        # Balance exploration (random search) and exploitation (local search)
        if np.random.random() < exploration_rate:  # Exploration with adaptive rate
            current_point = np.random.uniform(min_bound, max_bound, size=2)
        else:  # Exploitation - search near current best with adaptive step size
            step_size = 0.5 * (1 - i/iterations)  # Gradually reduce step size
            current_point = np.clip(best_point + np.random.normal(0, step_size, size=2), 
                                   min_bound, max_bound)
            
        current_value = evaluate_function(*current_point)

        # Update best if current point is better
        if current_value < best_value:
            best_point = current_point
            best_value = current_value
            recent_improvements += 1
        else:
            recent_improvements = max(0, recent_improvements - 1)

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