import numpy as np
import matplotlib.pyplot as plt
# Then we take the parameter value lists, build the state vectors corresponding
# to each circuit, and plot them on the Bloch sphere:
from qiskit.visualization.bloch import Bloch
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit, Parameter

# script used to generate Bloch spheres only for visualization purposes
theta_param = Parameter('θ')
phi_param = Parameter('Φ')
lambda_param = Parameter('λ')

# Circuit A
qc_A = QuantumCircuit(1)
qc_A.h(0)
qc_A.rz(theta_param, 0)
qc_A.draw(output='mpl')
plt.savefig('img/mapping_example/circuitA')

# Circuit B
qc_B = QuantumCircuit(1)
qc_B.h(0)
qc_B.rz(theta_param, 0)
qc_B.rx(phi_param, 0)
qc_B.draw(output='mpl')
plt.savefig('img/mapping_example/circuitB')

# Circuit C
qc_C = QuantumCircuit(1)
qc_C.h(0)
qc_C.rz(theta_param, 0)
qc_C.rx(phi_param, 0)
qc_C.ry(lambda_param, 0)
qc_C.draw(output='mpl')
plt.savefig('img/mapping_example/circuitC')

# Next we uniformly sample the parameter space for the two parameters theta and phi
np.random.seed(0)
num_param = 1000
theta = [2 * np.pi * np.random.uniform() for i in range(num_param)]
phi = [2 * np.pi * np.random.uniform() for i in range(num_param)]
lambda_p = [2 * np.pi * np.random.uniform() for i in range(num_param)]


def state_to_bloch(state_vec):
    # Converts state vectors to points on the Bloch sphere
    phi = np.angle(state_vec.data[1]) - np.angle(state_vec.data[0])
    theta = 2 * np.arccos(np.abs(state_vec.data[0]))
    return [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]


# Bloch sphere plot formatting
width, height = plt.figaspect(1 / 1)
fig1, fig2, fig3 = plt.figure(figsize=(width, height)),\
    plt.figure(figsize=(width, height)),\
    plt.figure(figsize=(width, height))
ax1, ax2, ax3 = fig1.add_subplot(projection='3d'),\
    fig2.add_subplot(projection='3d'),\
    fig3.add_subplot(projection='3d')
b1, b2, b3 = Bloch(axes=ax1), Bloch(axes=ax2), Bloch(axes=ax3)
b1.point_color, b2.point_color, b3.point_color = ['tab:blue'], ['tab:blue'], ['tab:blue']
b1.point_marker, b2.point_marker, b3.point_marker = ['o'], ['o'], ['o']
b1.point_size, b2.point_size, b3.point_size = [2], [2], [2]

# Calculate state vectors for circuit A and circuit B for each set of sampled parameters
# and add to their respective Bloch sphere
for i in range(num_param):
    state_1 = Statevector.from_instruction(qc_A.bind_parameters({theta_param: theta[i]}))
    state_2 = Statevector.from_instruction(qc_B.bind_parameters({theta_param: theta[i], phi_param: phi[i]}))
    state_3 = Statevector.from_instruction(qc_C.bind_parameters({theta_param: theta[i],
                                                                 phi_param: phi[i],
                                                                 lambda_param: lambda_p[i]}))
    b1.add_points(state_to_bloch(state_1))
    b2.add_points(state_to_bloch(state_2))
    b3.add_points(state_to_bloch(state_3))

b1.show()
fig1.savefig('img/mapping_example/mappingA')
b2.show()
fig2.savefig('img/mapping_example/mappingB')
b3.show()
fig3.savefig('img/mapping_example/mappingC')
