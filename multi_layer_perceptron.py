import numpy as np
import random

class Multi_layer_perceptron:
    def __int__(self):
        self.neuron_1 = None
        self.neuron_2 = None


class Neuron:
    def __int__(self):
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.w = None
        self.weighted_sum = None
        self.output = None
        # one neuron in our case because it is of output layer(a single neuron)
        self.next_layer_neuron = None

    def weighted_sum_(self, x1, x2, x3, weights):
        return (x1 * weights[0]) + (x2 * weights[1]) + (x3 * weights[2])

    def tanh_activation(self, weighted_sum):
        return (np.exp(weighted_sum) - np.exp(-weighted_sum)) / (np.exp(weighted_sum) + np.exp(-weighted_sum))

    def step_activation(self, weighted_sum):
        if weighted_sum >= 0:
            return 1
        else:
            return -1


N1 = Neuron()
N2 = Neuron()
N3 = Neuron()
N1.next_layer_neuron = N3
N2.next_layer_neuron = N3
MLP_1 = Multi_layer_perceptron()
MLP_1.neuron_1 = N1
MLP_1.neuron_2 = N2
# print(MLP_1.neuron_2.next_layer_neuron)
trainInput = [[1, 1], [9.4, 6.4], [2.5, 2.1], [8, 7.7], [0.5, 2.2],
              [7.9, 8.4], [7, 7], [2.8, 0.8], [1.2, 3], [7.8, 6.1]]

trainOutput = [1, -1, 1, -1, -1, 1, -1, 1, -1, -1]


def population_generation(population_size):
    population = []
    for _ in range(population_size):
        population_individual = []
        for _ in range(3):
            weight_vector = []
            for _ in range(3):
                weight_vector.append(np.random.uniform(-1, 1))
            population_individual.append(weight_vector)
        population.append(population_individual)
    return population

def fitness_evaluation(current_population):
    fitness_values = {}
    weight_index = 0
    for individual in current_population:
        individual_correctness = 0
        for i in range(len(trainInput)):
            MLP_1.neuron_1.x1 = trainInput[i][0]
            MLP_1.neuron_1.x2 = trainInput[i][1]
            MLP_1.neuron_1.x3 = 1
            MLP_1.neuron_1.w = individual[0]

            MLP_1.neuron_2.x1 = trainInput[i][0]
            MLP_1.neuron_2.x2 = trainInput[i][1]
            MLP_1.neuron_2.x3 = 1
            MLP_1.neuron_2.w = individual[1]

            MLP_1.neuron_1.weighted_sum = MLP_1.neuron_1.weighted_sum_(
                MLP_1.neuron_1.x1, MLP_1.neuron_1.x2, MLP_1.neuron_1.x3, MLP_1.neuron_1.w)
            MLP_1.neuron_2.weighted_sum = MLP_1.neuron_2.weighted_sum_(
                MLP_1.neuron_2.x1, MLP_1.neuron_2.x2, MLP_1.neuron_2.x3, MLP_1.neuron_2.w)

            MLP_1.neuron_1.output = MLP_1.neuron_1.tanh_activation(MLP_1.neuron_1.weighted_sum)
            MLP_1.neuron_2.output = MLP_1.neuron_2.tanh_activation(MLP_1.neuron_2.weighted_sum)

            MLP_1.neuron_1.next_layer_neuron.x1 = MLP_1.neuron_1.output
            MLP_1.neuron_1.next_layer_neuron.x2 = MLP_1.neuron_2.output
            MLP_1.neuron_1.next_layer_neuron.x3 = 1
            MLP_1.neuron_1.next_layer_neuron.w = individual[2]

            MLP_1.neuron_1.next_layer_neuron.weighted_sum = MLP_1.neuron_1.next_layer_neuron.weighted_sum_(
                MLP_1.neuron_1.next_layer_neuron.x1, MLP_1.neuron_1.next_layer_neuron.x2,
                MLP_1.neuron_1.next_layer_neuron.x3, MLP_1.neuron_1.next_layer_neuron.w)

            MLP_1.neuron_1.next_layer_neuron.output = MLP_1.neuron_1.next_layer_neuron.step_activation(
                MLP_1.neuron_1.next_layer_neuron.weighted_sum)

            if MLP_1.neuron_1.next_layer_neuron.output == trainOutput[i]:
                individual_correctness = individual_correctness + 1

        fitness_values[weight_index] = individual_correctness * 10
        weight_index = weight_index + 1
    return fitness_values


def natural_selection(fitness_values):
    sorted_weights = sorted(fitness_values.items(), key=lambda x: x[1], reverse=True)
    sorted_weights = dict(sorted_weights)
    return sorted_weights


def next_generation(sorted_weights, current_population):
    new_generation = []
    sorted_weight_indexes = list(sorted_weights.keys())
    
    for i in range(0, len(sorted_weight_indexes), 2):
        if i == 0:
            new_generation.append(current_population[sorted_weight_indexes[i]])
            new_generation.append(current_population[sorted_weight_indexes[i + 1]])
        else:
            childs = crossover(current_population[sorted_weight_indexes[i]], current_population[sorted_weight_indexes[i + 1]])
            new_generation.append(childs[0])
            new_generation.append(childs[1])
    return new_generation


def crossover(parent1, parent2):
    parent1_1d_list = []
    parent2_1d_list = []
    p1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    p2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for row in range(3):
        for column in range(3):
            parent1_1d_list.append(parent1[column][row])
            parent2_1d_list.append(parent2[column][row])

    tooka = random.randint(0, 8)
    child1 = []
    child2 = []
    child1.extend(parent1_1d_list[:tooka])
    child1.extend(parent2_1d_list[tooka:])

    child2.extend(parent2_1d_list[:tooka])
    child2.extend(parent1_1d_list[tooka:])

    k = 0
    for row in range(3):
        for column in range(3):
            p1[column][row] = child1[k]
            p2[column][row] = child2[k]
            k = k + 1

    child = []
    child.append(p1)
    child.append(p2)

    return child


trained_weights = []


trained_weights_sets = []
trained_weights_1 = [[[0.21533906204659292, -0.7099824775961303, 0.6165797429997819], [-0.06362251787224826, -0.12830647832428554, 0.8567969595206613], [0.5026100161702169, -0.260925656731412, 0.34730592579002906]]]
trained_weights_2 = [[[-0.22936843917166194, 0.891580109209956, -0.8598767636702527], [-0.08442877695523543, 0.30486967962611966, -0.06150559664150901], [-0.7846438512718374, 0.5190146183422046, 0.29574267288040534]]]
trained_weights_3 = [[[-0.08249602669836209, 0.16810588786104064, 0.25458748472205883], [-0.4202291514291516, 0.8501522456609909, 0.027490128684435122], [0.6160569698384029, -0.6426674958676453, 0.18940942425350338]]]
trained_weights_4 = [[[0.3387055794758771, -0.5892919671663621, 0.945832732932042], [-0.4528550253285175, 0.8286601770559305, -0.9565773224131517], [-0.5136547625308385, -0.5875860769196715, 0.1402298378505944]]]

trained_weights_sets.append(trained_weights_1)
trained_weights_sets.append(trained_weights_2)
trained_weights_sets.append(trained_weights_3)
trained_weights_sets.append(trained_weights_4)
for i in range(4):
    fitness = fitness_evaluation(trained_weights_sets[i])
    print(fitness)





