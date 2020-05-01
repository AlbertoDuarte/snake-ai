import random
import numpy as np
from utils import clear, SNAKE, FRUIT

class Game(object):
    def __init__(self):
        self.points = 0
        self.finished = False
        self.size = 15 # constant
        self.player = [[self.size//2, self.size//2], [self.size//2, self.size//2 -1], [self.size//2, self.size//2 -2], [self.size//2, self.size//2 -3], [self.size//2, self.size//2 -4]]
        self.fruit = [self.size//2, self.size//2 +1]
        # self.grid = np.zeros(shape=(self.size, self.size)

    def step(self, move):
        cur = self.player[0]
        prox = [cur[0]+move[0], cur[1]+move[1]]
        if not self.inBounds(prox) or self.isFinished() or prox in self.player:
            self.finished = True
            return

        self.player.insert(0, prox)
        if (prox == self.fruit):
            self.spawnFruit()
            self.points += 1
        else:
            self.player.pop()


    def spawnFruit(self):
        fruit = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        while(fruit in self.player):
            fruit = [random.randint(0, self.size-1), random.randint(0, self.size-1)]

        self.fruit = fruit

    def isFinished(self):
        return self.finished

    def inBounds(self, prox):
        return not (prox[0] < 0 or prox[0] >= self.size or prox[1] < 0 or prox[1] >= self.size)

    def getState(self):
        state = self.grid.flatten()
        state.append([self.x, self.y])

    def getPoints(self):
        return self.points

    def printGrid(self):
        clear()
        grid = np.zeros(shape=(self.size, self.size), dtype = np.int32)
        for x, y in self.player:
            grid[int(x)][int(y)] = SNAKE

        grid[self.fruit[0], self.fruit[1]] = FRUIT

        for x in range(self.size):
            for y in range(self.size):
                if(grid[x][y] == FRUIT):
                    print("X", end = "")
                elif(grid[x][y] == SNAKE):
                    print("*", end = ""),
                else:
                    print(".", end = ""),
            print("\n")

        # print(grid)
