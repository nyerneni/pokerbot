'''
File: qlearn.py
'''
import collections

NUM_TRACKED = 7

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")


class QLearningAlgorithm(RLAlgorithm):

    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):

        # on each (s,a,r,s'):
        # w = w - eta(Qopt(s,a;w) - (r+ gamma * Vopt(s'))) * featurevector(s,a)
        # where Vopt(s') = max over actions from s' of Qopt(s', a')
        if not newState:
            return
        else:
            v_opt = max([self.getQ(newState, act) for act in self.actions(newState)])

        q_opt = self.getQ(state, action)
        factor = self.getStepSize() * (q_opt - (reward + self.discount * v_opt))
        for f, v in self.featureExtractor(state, action):
            update = factor * v
            self.weights[f] -= update


def pokerFeatureExtractor(state, action):
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
    isEnd = check_end_state(players)

    # features that need to be processed
    odds = get_win_prob(state)
    opponent_history = players[1].round_action_histories
    agent_stack = players[0].stack
    # opp_stack = players[1].stack
    pot_size = players[0].pay_info.amount + players[1].pay_info.amount

    # processed features and feature vector
    feature_pairs = []
    feature_pairs.append(((street, action_name), 1))
    pot_feature = bucket_pot(agent_stack, pot_size)
    feature_pairs.append(((pot_feature, action_name), 1))
    history_feature = process_history(opponent_history)
    feature_pairs.append(((history_feature, action_name), 1))

    if not isEnd:
        odds_feature = bucket_odds(odds)
        feature_pairs.append(((odds_feature, action_name), 1))

    return feature_pairs


def bucket_odds(odds):
    if odds <= .1:
        return "odds10"
    elif odds <= .2:
        return "odds20"
    elif odds <= .3:
        return "odds30"
    elif odds <= .4:
        return "odds40"
    elif odds <= .5:
        return "odds50"
    elif odds <= .6:
        return "odds60"
    elif odds <= .7:
        return "odds70"
    elif odds <= .8:
        return "odds80"
    elif odds <= .9:
        return "odds90"
    else:
        return "instantwin"


def process_history(history):
    temp_histories = list(reversed(history[-1:-8:-1]))
    actions = [entry["action"] for entry in temp_histories]
    while len(actions) < NUM_TRACKED:
        actions.append(None)
    counts = collections.defaultdict(int)
    majority_action, count = None, 0
    for action in actions:
        counts[action] += 1
        if counts[action] > count:
            count = counts[action]
            majority_action = action
    return majority_action


def bucket_pot(pot):
    if not stack: return "nopot"

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


def check_end_state(players):
    '''
    End state if either player has 0 and sthe river has ended, or a player has folded
    '''
    stack1, stack2 = players[0].stack, players[1].stack
    actions1, actions2 = players[0].action_histories, players[1].action_histories

    return (not stack1 or not stack2 or \
        actions1[-1]["action"] == "fold" or actions2[-1]["action"] == "fold")


# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards
