from os import name, system
GRID_SIZE = 10
SNAKE = 1
FRUIT = 2

MOVES_DICT = {0: 'W', 1: 'A', 2: 'S', 3: 'D'}
ACTIONS = {"W": (-1, 0), "S": (1, 0), "D": (0, 1), "A": (0, -1)}

def clear():
    if name == "nt":
        _ = system("cls")
    else:
        _ = system("clear")
