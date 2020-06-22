from game import Game
from utils import ACTIONS

game = Game()
game.printGrid()

while(not game.isFinished()):
    movestr = input()
    if(movestr.upper() in ACTIONS):
        move = ACTIONS[movestr.upper()]
        game.step(move)

    game.printGrid()
    print(game.getState())

print("You Lose!\nScore: {}".format(game.getPoints()))
