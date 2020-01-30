from game import Game
import random

game = Game()

while not game.over:
    moves = game.moves
    move = random.choice(moves)
    game.play(move)
    print(game)