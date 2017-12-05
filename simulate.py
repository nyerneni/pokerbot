from pypokerengine.api.game import start_poker, setup_config

from callbot import FishPlayer
from tightplayer import TightPlayer
from looseplayer import LoosePlayer
from ai import AIPlayer
import numpy as np
import pickle

if __name__ == '__main__':
    # The stack log contains the stacks of the Data Blogger bot after each game (the initial stack is 100)
    stack_log = []
    p1, p2 = TightPlayer(), AIPlayer()

    # with open("tightPlayer.weights", "rb") as f:
    #     weights = pickle.load(f)

    # p2.q.weights = weights
    num_learning_games = 30
    for x in range(num_learning_games):
        config = setup_config(max_round=100, initial_stack=500, small_blind_amount=25)
        config.register_player(name="p1", algorithm=p1)
        config.register_player(name="p2", algorithm=p2)
        game_result = start_poker(config, verbose=0)
        print game_result['players']
        if x:
            p2.q.explorationProb = 1.0/x

    p2.q.explorationProb = 0
    p2.learning = False
    print p2.q.numIters
    print "Done learning (maybe) after playing %d games" % num_learning_games

    print p2.q.weights
    with open("tightPlayer.weights", "wb+") as file:
        pickle.dump(p2.q.weights, file)


    num_wins = 0
    num_test_games = 10
    for x in range(num_test_games):
        config = setup_config(max_round=100, initial_stack=500, small_blind_amount=25)
        config.register_player(name="p1", algorithm=p1)
        config.register_player(name="p2", algorithm=p2)
        game_result = start_poker(config, verbose=0)
        if game_result['players'][1]['state'] != "folded": num_wins += 1
        print game_result['players']
    print "Win rate: {} (out of {} games)".format(float(num_wins)/num_test_games, num_test_games)
