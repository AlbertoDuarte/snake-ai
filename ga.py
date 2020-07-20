import pickle
import random
import numpy as np
from utils import GRID_SIZE, ACTIONS, MOVES_DICT
from nn import NeuralNetwork, Dense, ReLU, Sigmoid, Softmax
from game import Game
from tqdm import tqdm
from time import sleep

INPUT_SIZE = 24
MUT_CHANCE = 0.05
ETA = 50
ITERATIONS = 10000
POP_SIZE = 500

# print(random())

def createNN():
    # 24 X 20 X 12 X 4 feed forward neural network
    network = NeuralNetwork()
    network.addLayer(Dense(INPUT_SIZE, 20))
    network.addLayer(ReLU())
    network.addLayer(Dense(20, 12))
    network.addLayer(ReLU())
    network.addLayer(Dense(12, 4))
    network.addLayer(Softmax())
    return network

def crossover(matrix1, matrix2):
    # SBX crossover
    rand = np.random.random(matrix1.shape)
    gamma = np.empty(matrix1.shape)
    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (ETA + 1))  # First case of equation 9.11
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (ETA + 1))  # Second case

    offspring1 = 0.5 * ((1 + gamma)*matrix1 + (1 - gamma)*matrix2)
    offspring2 = 0.5 * ((1 - gamma)*matrix1 + (1 + gamma)*matrix2)

    return offspring1, offspring2

def mutate(matrix):
    # Gaussian mutation
    mutation = np.random.normal(size = matrix.shape)
    offspring = matrix + mutation
    return offspring

def selection(rank):
    # Roulette wheel selection
    parents = list()
    wheel = sum(r[0] for r in rank)
    for i in range(2):
        value = random.uniform(0, wheel)
        cur = 0
        for r in rank:
            cur += r[0]
            if cur > value:
                parents.append(r[1])
                break;

    return parents[0], parents[1]


def gen(population):
    # Generates new population
    rank = [[None, None, None] for i in range(POP_SIZE)]
    for i in range(POP_SIZE):
        nn = population[i]
        game = Game()
        points_last = 0
        moves_without_points = 0

        # Game loop
        while not game.isFinished() and moves_without_points < 2*(GRID_SIZE**2):
            state = game.getState()
            output = nn.calculate(state)[0]

            assert(len(output) == 4)

            # Chooses move with higher probability
            maior = output[0]
            move = 0
            for j in range(len(output)):
                val = output[j]
                if val > maior:
                    maior = val
                    move = j

            assert(move >= 0 and move <= 3)
            move = MOVES_DICT[move]

            game.step(ACTIONS[move])

            if game.getPoints() == points_last:
                moves_without_points += 1
            else:
                moves_without_points = 0
                points_last = game.getPoints()

        rank[i][0] = int(game.getReward())  # reward
        rank[i][1] = i                      # original index
        rank[i][2] = int(game.getPoints())  # fruits eaten

    rank.sort(key = lambda x: x[0], reverse = True)

    new_population = [None for x in range(POP_SIZE)]
    # crossover
    for i in range(0,POP_SIZE)[::2]:
        offspring1 = createNN()
        offspring2 = createNN()
        ind1, ind2 = selection(rank)
        network1, network2 = population[ind1], population[ind2]

        for l1, l2, off1, off2 in zip(network1.layers, network2.layers, offspring1.layers, offspring2.layers):
            if not l1.isWeighted() or not l2.isWeighted:
                continue
            new_w1, new_w2 = crossover(l1.getWeights(), l2.getWeights())
            new_b1, new_b2 = crossover(l1.getBias(), l2.getBias())

            off1.setWeights(new_w1), off2.setWeights(new_w2)
            off1.setBias(new_b1), off2.setBias(new_b2)

        new_population[i] = offspring1
        new_population[i+1] = offspring2

    # mutate
    for offspring in new_population:
        for new_l in offspring.layers:
            if not new_l.isWeighted():
                continue

            new_weight = mutate(new_l.getWeights())
            new_bias = mutate(new_l.getBias())

            new_l.setWeights(new_weight)
            new_l.setBias(new_bias)

    return new_population, rank[0][0], rank[0][2], population[rank[0][1]]


def geneticAlgo():
    population = list()
    for i in range(POP_SIZE):
        network = createNN()
        population.append(network)

    for i in tqdm(range(ITERATIONS)):
        population, best_reward, best_points, best = gen(population)
        if(i%100 == 0):
            print("best of {} has {} reward and {} points".format(i, best_reward, best_points))
            name = "saved/best-{}.pickle".format(i)
            with open(name, "wb+") as f:
                pickle.dump(best, f)

if __name__ == "__main__":
    geneticAlgo()
