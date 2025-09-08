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

    for _ in range(iterations):
        # Simple random search
        # Generate candidate point more concisely
        x, y = np.random.uniform(bounds[0], bounds[1], size=2)
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y
            
            # Local search around the new best point with decreasing step size
            step_size = (bounds[1] - bounds[0]) / 10  # Initial step size
            for _ in range(5):  # Limited local search iterations
                # Explore nearby points
                dx, dy = np.random.normal(0, step_size, size=2)
                nx, ny = best_x + dx, best_y + dy
                
                # Keep within bounds
                nx = np.clip(nx, bounds[0], bounds[1])
                ny = np.clip(ny, bounds[0], bounds[1])
                
                new_value = evaluate_function(nx, ny)
                if new_value < best_value:
                    best_value = new_value
                    best_x, best_y = nx, ny
                
                # Reduce step size for finer search
                step_size *= 0.5

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
