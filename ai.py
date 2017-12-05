from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.action_checker import ActionChecker
import qlearn
import collections
import pickle
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck
NB_SIMULATION = 1000

class AIPlayer(BasePokerPlayer):

    def __init__(self):
        self.q = qlearn.QLearningAlgorithm(actions = self.generate_possible_actions, \
                        discount = 1, featureExtractor=qlearn.pokerFeatureExtractor, \
                        explorationProb=1)
        self.sars = []
        self.prev = 100
        self.learning = True
        with open("tightPlayer.weights", "rb") as f:
            self.q.weights = pickle.load(f)

    def declare_action(self, valid_actions, hole_card, round_state):
        game_state = self.setup_game_state(round_state, hole_card)
        action = self.q.getAction(game_state, self.uuid)
        amount = self.get_amount(action["action"],valid_actions)
        self.sars = [game_state, action]
        return action['action'], amount

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        if self.learning:
            if round_state["street"] != "preflop":
                hole_card = self.sars[0]["table"].seats.players[0].hole_card
                prob = estimate_hole_card_win_rate(nb_simulation = 1000, nb_player=2, hole_card = hole_card,\
                                                     community_card = self.sars[0]["table"]._community_card)
                game_state = self.setup_game_state(round_state, [str(x) for x in hole_card])
                r = round_state["pot"]["main"]["amount"]
                self.q.incorporateFeedback(self.sars[0], self.sars[1], r*(2*prob - 1), game_state, self.uuid)


    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        #if len(self.sars) != 0:
        #    game_state = self.setup_game_state(round_state, hand_info)
        #    pot = round_state["pot"]["main"]["amount"]
        #    if self.uuid == winners[0]['uuid']:
        #        r = pot
        #    else:
        #        r = -pot/2
        #    self.q.incorporateFeedback(self.sars[0], self.sars[1], r, game_state)
        pass

    def generate_possible_actions(self, game_state):
        players = game_state["table"].seats.players
        player_pos = game_state["next_player"]
        sb_amount = game_state["small_blind_amount"]
        if player_pos == "not_found": return ActionChecker.legal_actions(players, 0, sb_amount)
        return ActionChecker.legal_actions(players, player_pos, sb_amount)

    def setup_game_state(self, round_state, my_hole_card):
        game_state = restore_game_state(round_state)
        for player in game_state["table"].seats.players:
            if player.uuid == self.uuid and my_hole_card != []:
                # Hole card of my player should be fixed. Because we know it.
                game_state = attach_hole_card(game_state, player.uuid, gen_cards(my_hole_card))
            else:
                # We don't know hole card of opponents. So attach them at random from deck.
                game_state = attach_hole_card_from_deck(game_state, player.uuid)
        return game_state

    def get_amount(self, action, valid_actions):

        if action == "raise":
            if valid_actions[2]["amount"]["max"] > 150:
                amount = 150
            elif valid_actions[2]["amount"]["max"] > 25:
                amount = 25
            else: amount = valid_actions[2]["amount"]["min"]
        elif action == "call":
            amount = valid_actions[1]["amount"]
        else:
            amount = 0
        return amount

def setup_ai():
    return AIPlayer()
