from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

import pickle
import os
from time import time

import numpy as np

import matplotlib.pyplot as plt

from generate_circuit import generate_circuit_2local, generate_circuit_paper, generate_circuit_2local_swap


class GeneticAlgorithm:
    def __init__(self, w, mode, num_qubits, num_features, num_genes, gene_length, pop_size, pool_size,
                 offspring_size, prob, mutation_prob, num_generations, early_stop):
        self.w = w
        self.mode = mode
        self.num_qubits = num_qubits
        self.num_features = num_features
        self.num_genes = num_genes
        self.gene_length = gene_length
        self.pop_size = pop_size
        self.pool_size = pool_size
        self.offspring_size = offspring_size
        self.prob = prob
        self.mutation_prob = mutation_prob
        self.num_generations = num_generations
        self.early_stop = early_stop

    def cost_func(self, acc, r, h, cnot, swap=0):
        if acc == 0.0:
            acc = 0.01
        gate_cost = r + 2 * h + 5 * cnot + 11 * swap
        fit = gate_cost + self.w / acc ** 2 - self.w
        return round(fit, 1)

    def calculate_fitness(self, pop, population, train_x, train_y, test_x, test_y):
        fitness, scores, qc_r, qc_h, qc_cnot = [], [], [], [], []

        for i in range(population):
            qc_i, r_i, h_i, cnot_i = None, None, None, None

            print(pop[i])

            if self.mode == 'paper':
                qc_i, r_i, h_i, cnot_i = generate_circuit_paper(pop[i], self.num_qubits, self.num_features)
            elif self.mode == '2local':
                qc_i, r_i, h_i, cnot_i = generate_circuit_2local(pop[i], self.num_qubits, self.num_features)
            elif self.mode == '2local_swap':
                qc_i, r_i, h_i, cnot_i, swap_i = generate_circuit_2local_swap(pop[i], self.num_qubits,
                                                                              self.num_features)

            kernel = FidelityQuantumKernel(feature_map=qc_i)
            model = SVC(kernel=kernel.evaluate)
            model.fit(train_x, train_y)
            score = model.score(test_x, test_y)

            print(score)

            scores.append(score)
            qc_r.append(r_i)
            qc_h.append(h_i)
            qc_cnot.append(cnot_i)

        for i in range(population):
            fit = self.cost_func(scores[i], qc_r[i], qc_h[i], qc_cnot[i], self.w)
            fitness.append(fit)

        return fitness, scores, qc_r, qc_h, qc_cnot

    def pareto_front(self, t_list):
        pareto_index = []
        tt_list = t_list.copy()

        for j in range(self.pop_size):
            min_pareto, min_index = [tt_list[0][0], tt_list[0][1]], 0

            for i in range(self.pop_size - j):
                if tt_list[i][0] >= min_pareto[0] and tt_list[i][1] <= min_pareto[1]:
                    min_pareto, min_index = [tt_list[i][0], tt_list[i][1]], i

            copy_index = [i for i, val in enumerate(t_list) if val == min_pareto]
            for r in copy_index:
                pareto_index.append(r)

            del tt_list[min_index]

        return pareto_index

    def init_population(self):
        if os.path.exists('population/parents.pkl'):
            with open('population/pop.pkl', 'rb') as file:
                pop = pickle.load(file)
        else:
            pop = []
            for i in range(self.pop_size):
                genes = np.random.randint(2, size=(self.num_genes, self.gene_length))
                pop.append(genes)

        return pop

    def execute(self, early_stop, threshold, train_x, train_y, test_x, test_y):
        cost, obj_gate_list, obj_acc_list = [], [], []

        pop = self.init_population()

        start_time = time()
        for g in range(self.num_generations):

            fitness, score, qc_r, qc_h, qc_cnot, qc_swap, plt_acc, plt_gate = [], [], [], [], [], [], [], []
            cost_pool, obj_gate, obj_acc = 0, 0, 0

            score, fitness, qc_r, qc_h, qc_cnot = self.calculate_fitness(pop, self.pop_size,
                                                                         train_x, train_y, test_x, test_y)
            end_time = time()

            for i in range(self.pool_size):
                cost_p = fitness[i]
                obj_g = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i])
                obj_a = score[i]

                plt_gate.append(obj_g)
                plt_acc.append(obj_a)
                cost_pool += cost_p
                obj_gate += obj_g
                obj_acc += obj_a

            if cost_pool / self.pool_size < threshold:
                threshold = cost_pool / self.pool_size
                early_stop = 0
            early_stop += 1

            cost.append(cost_pool / self.pool_size)
            obj_gate_list.append(obj_gate / self.pool_size)
            obj_acc_list.append(obj_acc / self.pool_size)
            plt.scatter(plt_acc, plt_gate, s=10, c="#4863A0",
                        alpha=(g + self.num_generations / 2) / (1.5 * self.num_generations))

            print('\nGeneration:', g + 1, ', Cost:', round(cost[g], 2), ', Time:', round(end_time - start_time, 2), 's')
            print('Accuracy:', score)
            print('Fitness:', fitness)

            if g == self.num_generations:
                break
            if early_stop == self.early_stop:
                break

            parents = []

            for i in range(self.pool_size):
                fitnessIndex = np.where(fitness == min(fitness))
                parents.append(pop[fitnessIndex[0][0]])
                del fitness[fitnessIndex[0][0]]
                del pop[fitnessIndex[0][0]]

            for i in range(self.offspring_size):
                ll, rr = np.random.randint(self.pool_size), np.random.randint(self.pool_size)
                parent_left, parent_right = parents[ll], parents[rr]
                cross_point = np.random.randint(self.num_genes - 1)
                offspring = np.concatenate((parent_left[:cross_point], parent_right[cross_point:]), axis=0)

                for ii in range(self.prob):
                    mutation_index = np.random.randint(self.num_genes)
                    mutation_bit = np.random.randint(self.gene_length)
                    offspring[mutation_index][mutation_bit] = (offspring[mutation_index][mutation_bit] + 1) % 2

                parents.append(offspring)

                # save the new population
                with open('population/pop.pkl', 'wb') as file:
                    pickle.dump(parents, file)

            pop = parents
        plt.show()

    @staticmethod
    def cost_gate(r, h, cnot, swap=0):
        return r + 2 * h + 5 * cnot + 11 * swap
