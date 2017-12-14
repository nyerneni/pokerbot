
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pokereval.hand_evaluator import HandEvaluator as HandEvaluator2
from pokereval.card import Card
import math
import random

def weightedsample(d):
    r = random.uniform(0, sum(d.itervalues()))
    s = 0.0
    for k, w in d.iteritems():
        s += w
        if r < s: return k
    return k

mandict = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J': 11, 'Q': 12, 'K': 13,
'A': 14}
res = dict((v,k) for k,v in mandict.iteritems())

lookuptable = {}

def builddistr(amount, discount, stack, hand):
    amount = min(amount, stack) - discount
    for i in range(2,15):
        for j in range(i,15):
            count = 4
            if hand[0] in [i,j]:
                count -= 1
            if hand[1] in [i,j]:
                count -= 1
            if j == hand
            hand = [Card(i, 1), Card(j, 1)]
            hand2 = [Card(i, 1), Card(j, 2)]
            val = math.exp(HandEvaluator2.evaluate_hand(hand, [])*(amount/100000)+5) * (count/4)
            val2 = math.exp(HandEvaluator2.evaluate_hand(hand2, [])*(amount/100000)+5) * (count/4)
            lookuptable[(i,j, 1)] = val
            lookuptable[(i,j, 0)] = val2


def simpreflop(hand, comm_card, amount, discount, stack):
    updthand = []
    updthand.append(mandict[hand[0][1:]])
    updthand.append(mandict[hand[1][1:]])
    oppdict = builddistr(amount, discount, stack, updthand)
    opp = weightedsample(oppdict)
    if opp[2]:
        opp = ['S'+str(res[opp[0]]), 'S'+str(res[opp[1]])]
    else:
        opp = ['S'+str(res[opp[0]]), 'H'+str(res[opp[1]])]
    comm_card = _fill_community_card(comm_card, hand + comm_card + opp)
    opp_score = HandEvaluator.eval_hand(opp, comm_card)
    player_score = HandEvaluator.eval_hand(hand, comm_card)
    return 1 if my_score >= max(opponents_score) else 0
