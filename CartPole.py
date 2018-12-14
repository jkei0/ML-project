"""My solution to open AI gym cartpole environment"""

import gym
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt


# Hyperparameters
EPISODES = 300
LEARNING_RATE = 0.001
EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.001
MEMORY_SIZE = 3000
BATCH_SIZE = 32
GAMMA = 0.95


class QNetwork:
    def __init__(self, act_space, obs_space):
        self.act_space = act_space
        self.obs_space = obs_space
        self.model = self.get_model()
        self.epsilon = EPSILON
        self.memory = []

    def get_model(self):
        # Build neural network
        model = keras.models.Sequential()

        # Input layer
        model.add(keras.layers.Dense(64,activation='relu', input_dim=OBS_SPACE))

        # First hidden layer
        model.add(keras.layers.Dense(32, activation='relu'))

        # Output layer
        model.add(keras.layers.Dense(self.act_space, activation='linear'))

        # Compile model
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        return model

    def take_action(self):
        action = 0
        # Take action using epsilon greedy policy
        if np.random.random() < self.epsilon:
            # Return 0 or 1 randomly
            action =  random.randrange(2)
        else:
            action = np.argmax(self.model.predict(state, batch_size=1))

        return action

    def remember(self, state, next_state, reward, done, action):

        # Remove first item
        if len(self.memory) >= MEMORY_SIZE:
            self.memory.pop(0)

        self.memory.append([state,next_state,reward,done,action])

    def train(self):

        if len(self.memory) < BATCH_SIZE:
            return

        # Get random samples from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([val[0] for val in batch])
        states = np.reshape(states,(BATCH_SIZE,4))
        next_states = np.array([(np.zeros(self.obs_space) if val[1] is None
                                else val[1]) for val in batch])
        next_states = np.reshape(next_states,(BATCH_SIZE,4))

        # Predict Q-values of state and next_state
        Qs = self.model.predict_on_batch(states)
        n_Qs = self.model.predict_on_batch(next_states)

        # Training arrays
        x = np.zeros((len(batch), OBS_SPACE))
        y = np.zeros((len(batch), ACT_SPACE))

        for i, b in enumerate(batch):
            state, next_state, reward, done, action = b[0], b[1], b[2],b[3], b[4]

            current_q = Qs[i]

            if done:
                current_q[action] = reward

            else:
                current_q[action] = reward + GAMMA * np.amax(n_Qs[i])

            x[i] = state
            y[i] = current_q

        self.model.train_on_batch(x,y)

        # calculate new epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon = self.epsilon * EPSILON_DECAY


def plot_rewards(reward, episode):
    plt.plot(reward, episode)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':

    rewards = []
    episodes = []

    # Create a environment
    env = gym.make('CartPole-v1')

    # Define action and observation spaces
    ACT_SPACE = env.action_space.n
    OBS_SPACE = env.observation_space.shape[0]

    # Create a QNetwork
    network = QNetwork(ACT_SPACE, OBS_SPACE)

    # Start playing game
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state,(1,4))

        for frame in range(500):

            # Decide action
            action = network.take_action()

            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 4))

            # Store action to memory
            network.remember(state,next_state,reward,done, action)

            # Train network
            network.train()

            state = next_state

            # If game ends
            if done:
                rewards.append(frame+1)
                episodes.append(episode)
                print("Episode {}/{} took {} frames to complete".format(episode, EPISODES, frame+1))
                break

    plot_rewards(episodes,rewards)
