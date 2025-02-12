import numpy as np
import matplotlib.pyplot as plt

def solve_heleshaw(nx, ny, dx, dy, u_top, u_bottom, u_left, u_right, max_iter=10000, tolerance=1e-6):
    """
    Solves the Hele-Shaw flow problem using the finite difference method.

    Args:
        nx, ny: Number of grid points in x and y directions.
        dx, dy: Grid spacing in x and y directions.
        u_top, u_bottom, u_left, u_right: Boundary conditions (velocities).
        max_iter: Maximum number of iterations.
        tolerance: Convergence tolerance.

    Returns:
        A tuple containing the velocity field (U, V) and the number of iterations.
    """
    # Initialize velocity field
    u = np.zeros((ny, nx))

    # Apply boundary conditions
    u[0, :] = u_bottom
    u[-1, :] = u_top
    u[:, 0] = u_left
    u[:, -1] = u_right

    # Iterative solution
    for iteration in range(max_iter):
        u_old = u.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

        # Check for convergence
        if np.max(np.abs(u - u_old)) < tolerance:
            break
    
    # Calculate velocity components (U, V) using central difference
    U = np.zeros((ny, nx))
    V = np.zeros((ny, nx))
    
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            U[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * dx)
            V[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dy)

    return (U, V), iteration + 1

# Example usage
nx = 50
ny = 50
dx = 0.1
dy = 0.1
u_top = 1.0
u_bottom = 0.0
u_left = 0.0
u_right = 0.0

(U, V), iterations = solve_heleshaw(nx, ny, dx, dy, u_top, u_bottom, u_left, u_right)

# Plotting the velocity field
x, y = np.meshgrid(np.arange(0, nx * dx, dx), np.arange(0, ny * dy, dy))
plt.figure(figsize=(8, 6))
plt.quiver(x, y, U, V, color='b')
plt.title(f'Hele-Shaw Flow Velocity Field (Iterations: {iterations})')
plt.xlabel('x')
plt.ylabel('y')
plt.show()