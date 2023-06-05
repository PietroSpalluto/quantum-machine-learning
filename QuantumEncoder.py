from qiskit import Aer, transpile
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt


class QuantumEncoder:
    def __init__(self, features, feature_map):
        self.features = features
        self.n_samples = features.shape[0]
        self.n_features = features.shape[1]
        self.feature_map = feature_map

        self.circuit = QuantumCircuit(self.n_features)

    # plots a scatter matrix highlighting a previously selected data point
    def plot_data_points(self, random_point):
        fig, axs = plt.subplots(self.n_features, self.n_features)
        fig.set_figwidth(3 * self.n_features)
        fig.set_figheight(3 * self.n_features)
        plt.rcParams.update({'font.size': 22})
        feature_names = [0, 1, 2, 3]
        for i in range(self.n_features):
            for j in range(self.n_features):
                if i == j:
                    axs[i][j].hist(self.features[:, j])
                else:
                    axs[i][j].scatter(self.features[:, j], self.features[:, i])
                    axs[i][j].scatter(self.features[:, j][random_point],
                                      self.features[:, i][random_point],
                                      color='r')
                    axs[i][j].set_aspect('equal', adjustable='datalim')
                if j == 0:
                    axs[i][j].set_ylabel(feature_names[i])
                if i == self.n_features - 1:
                    axs[i][j].set_xlabel(feature_names[j])

        plt.savefig('img/feature_map/data_points')

    # makes the circuit adding the feature map
    def make_circuit(self):
        self.circuit.compose(self.feature_map.decompose(), range(self.n_features), inplace=True)
        self.circuit.draw(output='mpl')
        plt.savefig('img/feature_map/feature_map')

    # encodes the classical data and obtains the corresponding state vectors
    def get_statevectors(self, random_point, meas=False):
        # obtains the statevectors of the encoded features
        state_vectors = self.encode()
        self.plot_state_vector(state_vectors, meas, random_point)

    # classical to quantum data encoding
    def encode(self):
        state_vectors = []

        backend = Aer.get_backend('statevector_simulator')
        # encodes every data point obtaining a state vector
        for i in range(len(self.features)):
            # print(features[i].tolist())
            # binding of the parameters (features) to the circuit
            encode = self.circuit.bind_parameters(self.features[i].tolist())

            job = backend.run(encode)
            result = job.result()
            outputstate = result.get_statevector(encode, decimals=3)
            state_vectors.append(outputstate)
            # print(outputstate)

        return state_vectors

    # adds a measurement to the circuit
    def add_measurement(self):
        meas = QuantumCircuit(self.n_features, self.n_features)
        meas.barrier(range(self.n_features))
        meas.measure(range(self.n_features), range(self.n_features))

        self.circuit.compose(meas, range(self.n_features), inplace=True)
        self.circuit.draw(output='mpl')
        plt.savefig('img/feature_map/measurement')

    # quantum circuit simulation
    def run_simulation(self):
        # backend for the quantum circuit simulator
        backend_sim = Aer.get_backend('qasm_simulator')

        # run simulation
        job_sim = backend_sim.run(transpile(self.circuit, backend_sim), shots=1024)
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        print(counts)

        plot_histogram(counts)
        plt.savefig('img/feature_map/sim_result')

    # saves the bloch sphere representation of the state vector associated to the previously
    # selected random point
    @staticmethod
    def plot_state_vector(state_vectors, meas, random_point):
        state_vectors[random_point].draw(output='bloch')
        if meas:
            plt.savefig('img/feature_map/measured_state_vector')
        else:
            plt.savefig('img/feature_map/state_vector')
