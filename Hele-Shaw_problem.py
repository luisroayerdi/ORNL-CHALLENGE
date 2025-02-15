'''
This script is an extension of the HHL solver, which is used to solve the linear system of equations
A * x = b, where A is a matrix, x is the solution vector, and b is the right-hand side vector.
The script generates a quantum circuit using Qiskit's HHL algorithm to solve this system for the Hele-Shaw flow problem.

The script first sets up a matrix and a vector from the `func_matrix_vector` module. 
These are customized for the Hele-Shaw problem (fluid dynamics in porous media).
The matrix (A) represents the system of equations, and the vector (b) contains the initial conditions or inputs to the system.

The `HHL` solver is then used to construct a quantum circuit, which is run on an ideal quantum simulator backend 
provided by Qiskit's Aer library. The solver uses quantum algorithms to find an approximate solution to the system of equations.

After constructing the circuit, the quantum solution is processed to extract the values of the vector x (solution),
representing the pressure and velocity fields of the Hele-Shaw problem.

The pressure solution is calculated both from the quantum solution (using the HHL solver) 
and from the analytical solution (using classical methods).

The velocity field is derived from the pressure field, and the results are plotted for both pressure and velocity, 
allowing a comparison between the quantum (HHL solver) and classical (analytical) methods.

Additionally, the fidelity between the quantum and analytical solutions is computed, providing a measure of 
how well the quantum solution approximates the exact classical solution.'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import time

from qiskit_aer import AerSimulator
from linear_solvers import HHL
import func_matrix_vector as matvec

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case_name", type=str, default='hele-shaw', help="Problem case name.")
parser.add_argument("-casefile", "--case_variable_file", type=str, default='input_vars.yaml', help="YAML file with problem variables.")
parser.add_argument("--savedata", default=False, action='store_true', help="Save computed data.")
args = parser.parse_args()

if __name__ == '__main__':
    matrix, vector, input_vars = matvec.get_matrix_vector(args)
    MATRIX_SIZE = matrix.shape[0]

    backend = AerSimulator(method='statevector')
    print(f'Using AerSimulator with statevector backend')

    print(f'==================Solving with HHL================', flush=True)
    hhl = HHL(quantum_instance=backend)
    
    t = time.time()
    solution_hhl = hhl.solve(matrix, vector)
    t_hhl = time.time() - t
    quantum_solution = solution_hhl.state.real 
    print(f'Time elapsed for HHL solution: {t_hhl:.2f} sec')

    classical_solution = np.linalg.solve(matrix, vector)
    print(f'Classical Solution: {classical_solution}')

    error = np.linalg.norm(quantum_solution - classical_solution) / np.linalg.norm(classical_solution)
    fidelity = np.dot(quantum_solution, classical_solution) / (np.linalg.norm(quantum_solution) * np.linalg.norm(classical_solution))
    
    print(f'Error between HHL and analytical solution: {error:.6f}')
    print(f'Fidelity between solutions: {fidelity:.6f}')

    # Compute Velocity from Pressure**
    def compute_velocity(pressure):
        """Compute velocity as the gradient of the pressure field."""
        nx = int(np.sqrt(len(pressure)))  # Assuming a square domain
        pressure_field = pressure.reshape((nx, nx))
        vx, vy = np.gradient(-pressure_field)  # Velocity is negative gradient of pressure
        return vx, vy

    velocity_hhl_x, velocity_hhl_y = compute_velocity(quantum_solution)
    velocity_classical_x, velocity_classical_y = compute_velocity(classical_solution)

    # Plot Pressure Comparison
    plt.figure(figsize=(8, 5))
    plt.plot(classical_solution, 'bo-', label="Analytical Pressure")
    plt.plot(quantum_solution, 'ro-', label="Quantum HHL Pressure")
    plt.xlabel("Index")
    plt.ylabel("Pressure Value")
    plt.legend()
    plt.title("Comparison of Analytical vs. Quantum HHL Pressure")
    plt.grid()
    plt.savefig("pressure_comparison.png")
    plt.show()

    # Plot Velocity Fields
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].quiver(velocity_classical_x, velocity_classical_y, color='b', label="Analytical Velocity")
    ax[0].set_title("Analytical Velocity Field")
    
    ax[1].quiver(velocity_hhl_x, velocity_hhl_y, color='r', label="Quantum Velocity")
    ax[1].set_title("Quantum Velocity Field (HHL)")
    
    plt.savefig("velocity_fields.png")
    plt.show()

    # ** Save Results**
    if args.savedata:
        save_data = {
            'matrix': matrix,
            'vector': vector,
            'quantum_solution': quantum_solution,
            'classical_solution': classical_solution,
            'velocity_hhl_x': velocity_hhl_x,
            'velocity_hhl_y': velocity_hhl_y,
            'velocity_classical_x': velocity_classical_x,
            'velocity_classical_y': velocity_classical_y,
            'error': error,
            'fidelity': fidelity
        }
        with open("hele_shaw_results.pkl", "wb") as file:
            pickle.dump(save_data, file)
        print("===========Results saved===========")
