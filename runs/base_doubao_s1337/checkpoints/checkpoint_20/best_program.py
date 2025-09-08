# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""
import numpy as np


def search_algorithm(iterations: int = 1000, bounds: tuple[float, float] = (-5, 5)) -> tuple[float, float, float]:
    """
    Random search algorithm for function minimization.

    Args:
        iterations: Number of search iterations (default: 1000)
        bounds: Search space bounds as (min, max) (default: (-5, 5))

    Returns:
        Tuple containing (best_x, best_y, best_value)
    """
    def generate_random_point() -> tuple[float, float]:
        """Generate a random point within the search bounds"""
        return tuple(np.random.uniform(*bounds, size=2))
    
    # Initialize with random point
    best_x, best_y = generate_random_point()
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Generate new candidate point and evaluate
        x, y = generate_random_point()
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
