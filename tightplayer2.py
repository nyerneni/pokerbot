from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
NB_SIMULATION = 1000

class TightPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=NB_SIMULATION,
                nb_player=self.nb_player,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
                )
          # fetch CALL action info
        if 'call' in valid_actions[1]['action']:
            callbool = True
        else: callbool = False
        amount = valid_actions[1]['amount']
        action = valid_actions[0]
        if win_rate >= .85:
            action = valid_actions[1]
        elif win_rate >= .7:
            action = valid_actions[1]
        elif win_rate >= .55 :
            action = valid_actions[1]
        elif callbool == True and amount == 0:
            action = valid_actions[1]
        if amount < 0 or action['action'] == 'fold':
            amount = 0
        return action['action'], amount

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return HonestPlayer()
