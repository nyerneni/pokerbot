# pokerbot

## The following repo contains:

### looseplayer.py, tightplayer.py and callbot.py
- These are our hard coded poker bots that we trained against.

### ai.py
- This is our agent that implements the qlearning algorithm outlined in qlearn.py and the modified
Monte Carlo method in mctest.py

## neural.py
- When placed in the same directory as the training data, seperates the data into training and test datasets and trains the networa, reporting accuracy on the test set as well as saving the network. Also contains code to train over the entire dataset as well as tune hyperparameters using 5-fold cross validation
- Command: python neural.py

### simulate.py
- This is the script that needs to be run to simulate games between agents.
- Command required: python simulate.py
