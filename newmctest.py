
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
from pypokerengine.engine.hand_evaluator import HandEvaluator


def build_opp_range()

def sim(hand, comm_card):
    opp = sample_opp(2, hand + comm_card)
    comm_card = _fill_community_card(comm_card, hand + comm_card + opp)
    opp_score = HandEvaluator.eval_hand(opp, comm_card)
    player_score = HandEvaluator.eval_hand(hand, comm_card)
    return 1 if my_score >= max(opponents_score) else 0
