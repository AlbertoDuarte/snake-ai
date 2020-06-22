import random
import numpy as np
from utils import clear, SNAKE, FRUIT, GRID_SIZE

class Game(object):
    def __init__(self, seed = None):
        if(seed != None):
            random.seed(seed)
        self.moves = 0
        self.points = 0
        self.finished = False
        self.size = GRID_SIZE # constant
        self.head = [self.size//2, self.size//2]
        self.player = [[self.size//2, self.size//2], [self.size//2, self.size//2 -1], [self.size//2, self.size//2 -2]]
        # self.fruit = [self.size//2, self.size//2 +1]
        self.spawnFruit()

    def step(self, move):
        cur = self.player[0]
        prox = [cur[0]+move[0], cur[1]+move[1]]
        if not self.inBounds(prox) or self.isFinished() or prox in self.player:
            self.finished = True
            return

        self.moves += 1
        self.head = list(prox)
        self.player.insert(0, list(prox))
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

    def vision(self, dir):
        assert(len(dir) == 2)
        cur = [self.player[0][0]+dir[0], self.player[0][1]+dir[1]]
        body_found, fruit_found = False, False
        body_dist, fruit_dist = np.inf, np.inf

        dist = 1
        while (self.inBounds(cur)):
            if (not body_found and cur in self.player):
                body_found = True
                body_dist = dist

            if (not fruit_found and cur == self.fruit):
                fruit_found = True
                fruit_dist = dist

            dist+=1
            cur[0] += dir[0]
            cur[1] += dir[1]

        body_dist = 1.0/body_dist
        fruit_dist = 1.0/fruit_dist
        wall_dist = 1.0/dist

        return [fruit_dist, body_dist, wall_dist]

    def getState(self):
        state = []
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]] # Up, Down, Left, Right
        for dir in directions:
            state.extend(self.vision(dir))

        return state
        print(state)

    def getPoints(self):
        return self.moves + 100*(self.points**2)

    def getGrid(self, show_fruit = True):
        grid = np.zeros(shape=(self.size, self.size), dtype = np.int32)
        for x, y in self.player:
            grid[int(x)][int(y)] = SNAKE

        if (show_fruit):
            grid[self.fruit[0], self.fruit[1]] = FRUIT
        return grid

    def getMoves(self):
        return self.moves

    def printGrid(self):
        clear()
        grid = self.getGrid()

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
