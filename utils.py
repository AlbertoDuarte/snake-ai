from os import name, system
SNAKE = 1
FRUIT = 2

ACTIONS = {"W": (-1, 0), "S": (1, 0), "D": (0, 1), "A": (0, -1)}

def clear():
    if name == "nt":
        _ = system("cls")
    else:
        _ = system("clear")
