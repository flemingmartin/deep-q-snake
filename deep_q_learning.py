from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import random

import constants as const
from snake_env import SnakeEnv
from snake import Snake

# An episode a full game
train_episodes = 50_000
test_episodes = 100

channels = 1
input_dim = 20
output_dim = 4

save_model_every = 100

env = SnakeEnv(const.SCREEN_SIZE)




def create_model(input_dim,output_dim):
    learning_rate = 0.001

    input_layer = Input(shape=input_dim)
    hidden_layer = Dense(18, activation='relu')(input_layer)
    hidden_layer = Dense(18, activation='relu')(hidden_layer)
    output_layer = Dense(output_dim, activation='linear')(hidden_layer)

    model = Model(input_layer, output_layer)

    model.compile(loss='mse', optimizer=SGD(lr = learning_rate))

    return model


def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]


def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.95

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)



def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    # 1. Initialize the Target and Main models
    # Main Model (updated every step)
    model = create_model(input_dim,output_dim)
    # Target Model (updated every 100 steps)
    target_model = create_model(input_dim,output_dim)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0
    max_score = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        steps_whitout_reward = 0
        score = 0
        while not done:
            steps_to_update_target_model += 1
            if True:
                env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = np.random.randint(4)
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                #encoded = encode_observation(observation, env.observation_space.shape[0])
                encoded_reshaped = np.array(observation).reshape((1,)+ np.array(observation).shape)
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done = env.step(action+1)

            if reward == const.WIN_REWARD:
                steps_whitout_reward = 0
                score += 1
                if score > max_score:
                    max_score = score
            else:
                steps_whitout_reward += 1
                if steps_whitout_reward == const.MAX_STEPS_WHITOUT_REWARD:
                    done = True
                    reward = const.INFINITE_LOOP_REWARD
                    print('INFINITE LOOP')

            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 10 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Epoca: {} Score: {} MaxScore: {} AcumulatedReward: {}'.format(episode, score, max_score, total_training_rewards))
                total_training_rewards += 1

                if episode % save_model_every == 0:
                    print('Saving model...')
                    #model.save(f'/content/drive/MyDrive/Deep/Reinforcement Learning/Snake/Deep_Snake_V1/models3/model{episode}.h5')

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    #env.close()



if __name__ == '__main__':
    main()
