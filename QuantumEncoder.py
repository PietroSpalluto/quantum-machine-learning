from qiskit import Aer, QuantumCircuit

from qiskit.visualization import state_visualization
from qiskit.visualization.bloch import Bloch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# this class is used to map points from a n-dimensional space to a Hilbert space
class QuantumEncoder:
    def __init__(self, features, labels):
        self.features = features
        self.n_samples = features.shape[0]
        self.n_features = features.shape[1]
        self.n_classes = len(np.unique(labels))
        self.labels = labels

        self.state_vectors = []
        self.vectors_data = None

        self.figs = []
        self.axes = []
        self.spheres = []

    # classical to quantum data encoding
    def encode(self, circuit, var_params, measured=False):
        """
        Classical features are encoded to statevectors
        :param circuit: encoding circuit
        :param var_params: parameters of the variational circuit
        :param measured: flag indicating if a circuit has a measurement
        """
        self.state_vectors = []
        self.vectors_data = []

        # measurement are removed if we need to plot points before the measurement
        if measured:
            circuit.remove_final_measurements()

        backend = Aer.get_backend('statevector_simulator')
        # encodes every data point obtaining a state vector
        for i in range(len(self.features)):
            # binding of the parameters (features) to the circuit
            params = np.concatenate((self.features[i], var_params))
            encode = circuit.bind_parameters(params)

            job = backend.run(encode)
            result = job.result()
            outputstate = result.get_statevector(encode, decimals=3)
            self.state_vectors.append(outputstate)

        # statevectors are converted to be visualized in a Bloch sphere
        for i in range(len(self.state_vectors)):
            self.vectors_data.append(state_visualization._bloch_multivector_data(self.state_vectors[i]))

        self.vectors_data = pd.DataFrame(self.vectors_data)

    def init_plots(self):
        """
        3D Bloch spheres plot are initialized, one for each feature
        """
        self.figs = []
        self.axes = []
        self.spheres = []

        # Bloch sphere plot formatting
        width, height = plt.figaspect(1 / 1)
        for i in range(len(self.vectors_data.columns)):
            fig = plt.figure(figsize=(width, height))
            self.figs.append(fig)

        for i in range(len(self.vectors_data.columns)):
            axis = self.figs[i].add_subplot(projection='3d')
            self.axes.append(axis)

        colors = [['tab:blue'], ['tab:orange'], ['tab:green']]  # colors for three classes
        markers = [['o'], ['s'], ['d']]
        for i in range(len(self.vectors_data.columns)):
            class_spheres = []
            sphere_alpha = 0.2
            frame_alpha = 0.2
            xlabel = ['$x$', '']
            ylabel = ['$y$', '']
            zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
            for j, c, m in zip(range(self.n_classes), colors, markers):
                b = Bloch(axes=self.axes[i])
                b.sphere_alpha = sphere_alpha
                b.frame_alpha = frame_alpha
                b.xlabel = xlabel
                b.ylabel = ylabel
                b.zlabel = zlabel
                sphere_alpha = 0
                frame_alpha = 0
                xlabel = ['', '']
                ylabel = ['', '']
                zlabel = ['', '']
                b.point_marker = m
                b.point_size = [20]
                b.point_color = c
                class_spheres.append(b)

            self.spheres.append(class_spheres)

        # sphere for the random point that will be highlighted
        sphere_alpha = 0
        frame_alpha = 0
        xlabel = ['', '']
        ylabel = ['', '']
        zlabel = ['', '']
        for i in range(len(self.vectors_data.columns)):
            b = Bloch(axes=self.axes[i])
            b.sphere_alpha = sphere_alpha
            b.frame_alpha = frame_alpha
            b.xlabel = xlabel
            b.ylabel = ylabel
            b.zlabel = zlabel
            b.point_marker = ['o']
            b.point_size = [30]
            b.point_color = 'r'
            self.spheres[i].append(b)

    def add_data_points(self, random_point):
        """
        Add encoded data points to the bloch spheres
        :param random_point: highlighted point
        """
        last_sphere_idx = self.labels.max() + 1
        for i in range(len(self.spheres)):
            for j in range(len(self.vectors_data)):
                self.spheres[i][self.labels[j]].add_points(self.vectors_data[i].iloc[j])
                if j != random_point:
                    self.spheres[i][self.labels[j]].add_points(self.vectors_data[i].iloc[j])

            # the random point is added at the end, so it is above other points
            self.spheres[i][last_sphere_idx].add_points(self.vectors_data[i].iloc[random_point])

    def save_bloch_spheres(self, name):
        """
        Save images or multiple images used to animate the Bloch sphere
        :param name: name of the image
        """
        i = 0
        for c_spheres, fig, ax in zip(self.spheres, self.figs, self.axes):
            for s in c_spheres:
                s.show()
            # # for animation
            # for angle in range(360):
            #     ax.view_init(20, angle)
            #     fig.savefig('img/mapping/{} {}/ {}'.format(name, i, angle))

            # for bloch sphere plot
            fig.savefig('img/mapping/{} {}'.format(name, i))
            i += 1
        plt.clf()

    #
    def plot_data_points(self, random_point, feature_names, inverse_dict):
        """
        plots a scatter matrix highlighting a previously selected data point and using a different color
        for each class
        :param random_point: highlighted point
        :param feature_names: names of the dataset features
        :param inverse_dict: dictionary to convert label numbers into names
        """
        if self.n_features > 2:
            fig, axs = plt.subplots(self.n_features, self.n_features)
            fig.set_figwidth(3 * self.n_features)
            fig.set_figheight(3 * self.n_features)
            plt.rcParams.update({'font.size': 22})
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            for i in range(self.n_features):
                for j in range(self.n_features):
                    if i == j:
                        for k, c in zip(np.unique(self.labels), colors):
                            axs[i][j].hist(self.features[np.where(self.labels == k), j][0],
                                           color=c, label=inverse_dict[k], alpha=0.8)
                    else:
                        for k, c in zip(np.unique(self.labels), colors):
                            axs[i][j].scatter(self.features[np.where(self.labels == k), j],
                                              self.features[np.where(self.labels == k), i])
                        axs[i][j].scatter(self.features[:, j][random_point],
                                          self.features[:, i][random_point])
                        axs[i][j].set_aspect('equal', adjustable='datalim')
                    if j == 0:
                        axs[i][j].set_ylabel(feature_names[i])
                    if i == self.n_features - 1:
                        axs[i][j].set_xlabel(feature_names[j])

            axs[0][0].legend(bbox_to_anchor=(2*self.n_features/2, 1.5), loc='upper right',
                             ncols=self.n_features, fontsize=15)
        else:
            # plt.rcParams['figure.figsize'] = [6.4, 4.8]
            plt.figure(figsize=(6.4, 4.8))
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            for class_value in range(len(np.unique(self.labels))):
                # get row indexes for samples with this class
                row_ix = np.where(self.labels == class_value)
                # create scatter of these samples
                plt.scatter(self.features[row_ix, 0], self.features[row_ix, 1], c=colors[class_value],
                            label=inverse_dict[class_value])
            plt.scatter(self.features[random_point, 0], self.features[random_point, 1], c='r')
        plt.legend()
        plt.savefig('img/mapping/data_points')
        plt.clf()

    def add_measurement(self, circuit):
        """
        add a measurement to the circuit
        :param circuit: input circuit
        """
        meas = QuantumCircuit(self.n_features, self.n_features)
        meas.barrier(range(self.n_features))
        meas.measure(range(self.n_features), range(self.n_features))

        circuit.compose(meas, range(self.n_features), inplace=True)
        circuit.draw(output='mpl')
        plt.savefig('img/feature_map/measurement')
