# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations: int = 1000, bounds: tuple[float, float] = (-5, 5)) -> tuple[float, float, float]:
    """
    Adaptive random search algorithm for function minimization.
    
    Combines random exploration with occasional local refinement to balance
    between global exploration and local exploitation of promising regions.

    Args:
        iterations: Number of search iterations
        bounds: Tuple containing (min, max) values defining the search space

    Returns:
        Tuple of (best_x, best_y, best_value) representing the minimum found
    """
    # Initialize with random starting point within search bounds
    best_x, best_y = np.random.uniform(*bounds, size=2)
    best_val = evaluate_function(best_x, best_y)

    # Main search loop with adaptive strategy
    for i in range(iterations):
        # Adaptive search strategy: 80% exploration, 20% local refinement (similar to top performers)
        if np.random.rand() < 0.2:  # 20% chance of local exploration
            # Search around current best with decreasing step size (linear decay)
            range_size = bounds[1] - bounds[0]
            step = 0.1 * (1 - i/iterations) * range_size  # Linear decay like Program 1
            x = np.clip(best_x + np.random.normal(0, step), bounds[0], bounds[1])
            y = np.clip(best_y + np.random.normal(0, step), bounds[0], bounds[1])
        else:  # Random exploration
            x, y = np.random.uniform(bounds[0], bounds[1], size=2)
            
        value = evaluate_function(x, y)

        if value < best_val:
            best_val = value
            best_x, best_y = x, y

    return best_x, best_y, best_val


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
