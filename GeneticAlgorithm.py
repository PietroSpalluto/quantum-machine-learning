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
        fitness, scores, qc_r, qc_h, qc_cnot, qc_swap = [], [], [], [], [], []

        for i in range(population):
            qc_i, r_i, h_i, cnot_i, swap_i = None, None, None, None, None

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

            scores.append(score)
            qc_r.append(r_i)
            qc_h.append(h_i)
            qc_cnot.append(cnot_i)
            qc_swap.append(swap_i)

        for i in range(population):
            fit = self.cost_func(scores[i], qc_r[i], qc_h[i], qc_cnot[i])
            fitness.append(fit)

        return fitness, scores, qc_r, qc_h, qc_cnot, qc_swap

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
        if os.path.exists('genetic_algorithm/population.pkl'):
            with open('genetic_algorithm/population.pkl', 'rb') as file:
                pop = pickle.load(file)
        else:
            pop = []
            for i in range(self.pop_size):
                genes = np.random.randint(2, size=(self.num_genes, self.gene_length))
                pop.append(genes)

        return pop

    @staticmethod
    def load_stats():
        if os.path.exists('genetic_algorithm/statistics.pkl'):
            with open('genetic_algorithm/statistics.pkl', 'rb') as file:
                stats = pickle.load(file)
                gen_save = stats['Generation']
                plt_acc_save = stats['Plot Accuracy']
                plt_gate_save = stats['Plot Gate Complexity']
                fitness_save = stats['Fitness Values']
                score_save = stats['Scores']
                qc_r_save = stats['R Gates']
                qc_h_save = stats['H Gates']
                qc_cnot_save = stats['CNOT Gates']
                qc_swap_save = stats['SWAP Gates']
                cost_pool_save = stats['Cost']
                obj_gate_save = stats['Gate Complexity']
                obj_acc_save = stats['Accuracy']
                population_save = stats['Population']
                parents_save = stats['Parents']
                cost = stats['Mean Pool Fitness']
                obj_gate_list = stats['Mean Pool Gate Complexity']
                obj_acc_list = stats['Mean Pool Accuracy']
                pareto_save = stats['Pareto']

                print('Last generation statistics')
                print('\nGeneration:', gen_save[-1] + 1, ', Cost:', round(cost[gen_save[-1]], 2))
                print('Accuracy:', score_save[-1])
                print('Fitness:', fitness_save[-1])
        else:
            gen_save = []
            plt_acc_save = []
            plt_gate_save = []
            fitness_save = []
            score_save = []
            qc_r_save = []
            qc_h_save = []
            qc_cnot_save = []
            qc_swap_save = []
            cost_pool_save = []
            obj_gate_save = []
            obj_acc_save = []
            population_save = []
            parents_save = []
            cost = []
            obj_gate_list = []
            obj_acc_list = []
            pareto_save = []

        return (gen_save, plt_acc_save, plt_gate_save, fitness_save, score_save, qc_r_save, qc_h_save, qc_cnot_save,
                qc_swap_save, cost_pool_save, obj_gate_save, obj_acc_save, population_save, parents_save, cost,
                obj_gate_list, obj_acc_list, pareto_save)

    def execute(self, early_stop, threshold, train_x, train_y, test_x, test_y):
        # load statistics
        (gen_save, plt_acc_save, plt_gate_save, fitness_save, score_save, qc_r_save, qc_h_save, qc_cnot_save,
         qc_swap_save, cost_pool_save, obj_gate_save, obj_acc_save, population_save, parents_save, cost,
         obj_gate_list, obj_acc_list, pareto_save) = self.load_stats()

        pop = self.init_population()
        gen = gen_save[-1]

        start_time = time()
        for g in range(gen+1, self.num_generations):
            gen_save.append(g)
            population_save.append(pop)

            fitness, score, qc_r, qc_h, qc_cnot, qc_swap, plt_acc, plt_gate = [], [], [], [], [], [], [], []
            cost_pool, obj_gate, obj_acc = 0, 0, 0

            score, fitness, qc_r, qc_h, qc_cnot, qc_swap = self.calculate_fitness(pop, self.pop_size,
                                                                                  train_x, train_y,
                                                                                  test_x, test_y)

            fitness_save.append(fitness)
            score_save.append(score)
            qc_r_save.append(qc_r)
            qc_h_save.append(qc_h)
            qc_cnot_save.append(qc_cnot)
            qc_swap_save.append(qc_swap)

            end_time = time()

            for i in range(self.pool_size):
                cost_p = fitness[i]
                if self.mode == "2local_swap":
                    obj_g = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i], qc_swap[i])
                else:
                    obj_g = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i])
                obj_a = score[i]

                plt_gate.append(obj_g)
                plt_acc.append(obj_a)
                cost_pool += cost_p
                obj_gate += obj_g
                obj_acc += obj_a

            cost_pool_save.append(cost_pool)
            obj_gate_save.append(obj_gate)
            obj_acc_save.append(obj_acc)

            if cost_pool / self.pool_size < threshold:
                threshold = cost_pool / self.pool_size
                early_stop = 0
            early_stop += 1

            cost.append(cost_pool / self.pool_size)
            obj_gate_list.append(obj_gate / self.pool_size)
            obj_acc_list.append(obj_acc / self.pool_size)

            plt_acc_save.append(plt_acc)
            plt_gate_save.append(plt_gate)

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

            parents_save.append(parents)

            # save the new population
            with open('genetic_algorithm/population.pkl', 'wb') as file:
                pickle.dump(parents, file)

            # save algorithm statistics
            stats = {
                'Generation': gen_save,
                'Plot Accuracy': plt_acc_save,
                'Plot Gate Complexity': plt_gate_save,
                'Fitness Values': fitness_save,
                'Scores': score_save,
                'R Gates': qc_r_save,
                'H Gates': qc_h_save,
                'CNOT Gates': qc_cnot_save,
                'SWAP Gates': qc_swap_save,
                'Cost': cost_pool_save,
                'Gate Complexity': obj_gate_save,
                'Accuracy': obj_acc_save,
                'Population': population_save,
                'Parents': parents_save,
                'Mean Pool Fitness': cost,
                'Mean Pool Gate Complexity': obj_gate_list,
                'Mean Pool Accuracy': obj_acc_list,
                'Pareto': pareto_save
            }
            with open('genetic_algorithm/statistics.pkl', 'wb') as file:
                pickle.dump(stats, file)

            pop = parents
        plt.show()

    def execute_pareto_front(self, early_stop, threshold, train_x, train_y, test_x, test_y):
        # load statistics
        (gen_save, plt_acc_save, plt_gate_save, fitness_save, score_save, qc_r_save, qc_h_save, qc_cnot_save,
         qc_swap_save, _, obj_gate_save, obj_acc_save, population_save, parents_save, cost,
         obj_gate_list, obj_acc_list, pareto_save) = self.load_stats()

        pop = self.init_population()
        gen = gen_save[-1]

        start_time = time()
        for g in range(gen+1, self.num_generations):
            gen_save.append(g)
            population_save.append(pop)

            fitness, score, gate_cost, qc_r, qc_h, qc_cnot, qc_swap, plt_acc, plt_gate = ([], [], [], [], [], [], [],
                                                                                          [], [])
            cost_pool, obj_gate, obj_acc = 0, 0, 0

            score, _, qc_r, qc_h, qc_cnot, qc_swap = self.calculate_fitness(pop, self.pop_size,
                                                                            train_x, train_y,
                                                                            test_x, test_y)

            score_save.append(score)
            qc_r_save.append(qc_r)
            qc_h_save.append(qc_h)
            qc_cnot_save.append(qc_cnot)
            qc_swap_save.append(qc_swap)

            end_time = time()

            for i in range(self.pop_size):
                if self.mode == "2local_swap":
                    g_cost = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i], qc_swap[i])
                else:
                    g_cost = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i])
                gate_cost.append(g_cost)

            for i in range(self.pool_size):
                if self.mode == "2local_swap":
                    obj_g = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i], qc_swap[i])
                else:
                    obj_g = self.cost_gate(qc_r[i], qc_h[i], qc_cnot[i])
                obj_a = score[i]

                plt_gate.append(obj_g)
                plt_acc.append(obj_a)
                obj_gate += obj_g
                obj_acc += obj_a

            obj_gate_save.append(obj_gate)
            obj_acc_save.append(obj_acc)

            if score[0] > threshold[0] and gate_cost[0] < threshold[1]:
                threshold = [score[0], gate_cost[0]]
                early_stop = 0
            early_stop += 1

            obj_gate_list.append(obj_gate / self.pool_size)
            obj_acc_list.append(obj_acc / self.pool_size)

            plt_acc_save.append(plt_acc)
            plt_gate_save.append(plt_gate)

            plt.scatter(plt_acc, plt_gate, s=10, c="#4863A0",
                        alpha=(g + self.num_generations / 2) / (1.5 * self.num_generations))

            print('\nGeneration:', g + 1, ', Time:', round(end_time - start_time, 2), 's')
            print('Accuracy:', score)
            print('Fitness:', gate_cost)

            if g == self.num_generations:
                break
            if early_stop == self.early_stop:
                break

            parents, t_PARETO = [], []
            for i in range(self.pop_size):
                tt = [score[i], gate_cost[i]]
                t_PARETO.append(tt)

            pareto_save.append(t_PARETO)

            pareto_list = self.pareto_front(t_PARETO)
            for i in pareto_list:
                if i not in fitness:
                    fitness.append(i)

            fitness_save.append(fitness)

            for i in range(self.pool_size):
                FitnessIndex = fitness[i]
                parents.append(pop[FitnessIndex])

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

            parents_save.append(parents)

            # save the new population
            with open('genetic_algorithm/population.pkl', 'wb') as file:
                pickle.dump([parents, g], file)

            # save algorithm statistics
            stats = {
                'Generation': gen_save,
                'Plot Accuracy': plt_acc_save,
                'Plot Gate Complexity': plt_gate_save,
                'Fitness Values': fitness_save,
                'Scores': score_save,
                'R Gates': qc_r_save,
                'H Gates': qc_h_save,
                'CNOT Gates': qc_cnot_save,
                'SWAP Gates': qc_swap_save,
                'Gate Complexity': obj_gate_save,
                'Accuracy': obj_acc_save,
                'Population': population_save,
                'Parents': parents_save,
                'Mean Pool Fitness': cost,
                'Mean Pool Gate Complexity': obj_gate_list,
                'Mean Pool Accuracy': obj_acc_list,
                'Pareto': pareto_save
            }
            with open('genetic_algorithm/statistics.pkl', 'wb') as file:
                pickle.dump(stats, file)

            pop = parents
        plt.show()

    @staticmethod
    def cost_gate(r, h, cnot, swap=0):
        return r + 2 * h + 5 * cnot + 11 * swap
