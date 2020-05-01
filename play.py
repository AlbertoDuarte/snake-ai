from game import Game
from utils import ACTIONS

g = Game()
g.printGrid()

while(not g.isFinished()):
    movestr = input()
    if(movestr.upper() in ACTIONS):
        move = ACTIONS[movestr.upper()]
        g.step(move)

    g.printGrid()

print("You Lose!\nScore: {}".format(g.getPoints()))
