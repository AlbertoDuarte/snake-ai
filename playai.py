import pickle
from game import Game
from time import sleep
from utils import ACTIONS, MOVES_DICT

nn = None

try:
    dir = input("Enter AI pickle file: \n")
    nn = pickle.load(open(dir, "rb"))
except:
    print("error opening file")

game = Game()
game.printGrid()

while not game.isFinished():
    state = game.getState()
    output = nn.calculate(state)[0]

    assert(len(output) == 4)

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
    game.printGrid()
    print("output: {}".format(output))
    # print(game.getState())

    sleep(0.35)

if game.getWin():
    print("You win!")
else:
    print("You Lose!\nScore: {}".format(game.getPoints()))
input()
