'''
File: qlearn.py
'''
import math, random, copy
from collections import defaultdict
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import pypokerengine.engine.poker_constants as Const

NUM_TRACKED = 7

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState):
        eta = self.getStepSize()
        Q = self.getQ(state, action)
        if newState == None: return
        Vopt = max((self.getQ(newState,a)) for a in self.actions(newState))
        scale = eta*(Q - (reward + self.discount*Vopt))
        for key, value in self.featureExtractor(state, action):
            if key in self.weights:
                self.weights[key] -= scale*value

class QLearningAlgorithm(RLAlgorithm):

    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action, agentId):
        score = 0
        for f, v in self.featureExtractor(state, action, agentId):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state, agentId):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action, agentId), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState, agentId):

        # on each (s,a,r,s'):
        # w = w - eta(Qopt(s,a;w) - (r+ gamma * Vopt(s'))) * featurevector(s,a)
        # where Vopt(s') = max over actions from s' of Qopt(s', a')
        if not newState:
            return
        else:
            v_opt = max([self.getQ(newState, act, agentId) for act in self.actions(newState)])

        q_opt = self.getQ(state, action, agentId)
        factor = self.getStepSize() * (q_opt - (reward + self.discount * v_opt))
        for f, v in self.featureExtractor(state, action, agentId):
            update = factor * v
            self.weights[f] -= update


def pokerFeatureExtractor(state, action, agentId):
    '''
    TODO
    Ideas: indicators for stack size (buckets), opponent's last 5 actions,
            probability of hand winning (bucketed)
    '''

    """
    game_state = {
            "round_count": 0,
            "small_blind_amount": self.game_rule["sb_amount"],
            "street": Const.Street.PREFLOP,
            "next_player": None,
            "table": table
        }

    table:
        self.dealer_btn = 0
        self._blind_pos = None
        self.seats = Seats()
        self.deck = cheat_deck if cheat_deck else Deck()
        self._community_card = []

    to access players: table.seats.players (list of players)

    player:
        self.name = name
        self.uuid = uuid
        self.hole_card = []
        self.stack = initial_stack
        self.round_action_histories = self.__init_round_action_histories()
        self.action_histories = []
        self.pay_info = PayInfo()

    action = {"action": "fold/call/raise", "amount": 0}

    """

    # features that need no processing
    street = state["street"]
    action_name = action["action"]
    players = state["table"].seats.players
    agent = [player for player in players if player.uuid == agentId][0]
    opponent = [player for player in players if player.uuid != agentId][0]
    isEnd = check_end_state(players, street)

    # features that need to be processed
    odds = 1 if isEnd else get_win_prob(state["table"])
    opponent_history = opponent.round_action_histories
    # agent_stack = agent.stack
    # opp_stack = opponent.stack
    pot_size = players[0].pay_info.amount + players[1].pay_info.amount

    # processed features and feature vector
    feature_pairs = []
    feature_pairs.append(((street, action_name), 1))
    pot_feature = bucket_pot(pot_size)
    feature_pairs.append(((pot_feature, action_name), 1))
    history_feature = process_history(opponent_history)
    feature_pairs.append(((history_feature, action_name), 1))

    if not isEnd:
        odds_feature = bucket_odds(odds)
        feature_pairs.append(((odds_feature, action_name), 1))

    return feature_pairs

def get_win_prob(table):
    if table.seats.players[0].hole_card == []:
        return 0
    return estimate_hole_card_win_rate(nb_simulation = 1000, nb_player=2, hole_card = table.seats.players[0].hole_card, community_card = table._community_card)


def bucket_odds(odds):
    if odds <= .9:
        num = str((odds * 100) // 10)
        return "odds"+num
    else:
        return "instantwin"


def process_history(history):
    temp_histories = list(reversed(history[-1:-8:-1]))
    actions = [entry[0]["action"] for entry in temp_histories if entry is not None]
    while len(actions) < NUM_TRACKED:
        actions.append(None)
    counts = defaultdict(int)
    majority_action, count = None, 0
    for action in actions:
        counts[action] += 1
        if counts[action] > count:
            count = counts[action]
            majority_action = action
    return majority_action


def bucket_pot(pot):
    if pot <= 50:
        return "pot50"
    elif pot <= 100:
        return "pot100"
    elif pot <= 150:
        return "pot150"
    elif pot <= 250:
        return "pot250"
    elif pot <= 350:
        return "pot350"
    elif pot <= 500:
        return "pot500"
    elif pot <= 650:
        return "pot650"
    elif pot <= 800:
        return "pot800"
    else:
        return "bigkahuna"


def check_end_state(players, street):
    '''
    End state if either player has 0 and sthe river has ended, or a player has folded
    '''
    stack1, stack2 = players[0].stack, players[1].stack
    actions1, actions2 = players[0].action_histories, players[1].action_histories
    return ((stack1 == 0 and street == 5) or \
            (stack2 == 0 and street == 5) or \
            0 if len(actions1) == 0 else actions1[-1]["action"] == "fold" or
            0 if len(actions2) == 0 else actions2[-1]["action"] == "fold")
