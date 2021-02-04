from tensorflow.keras.models import load_model

import numpy as np
import time

import constants as const
from snake_env import SnakeEnv
from snake import Snake


channels = 1
input_dim = 20
output_dim = 4

model = load_model('models/model.h5')

env = SnakeEnv(const.SCREEN_SIZE)


MAX_STEPS_WHITOUT_REWARD = 100

while True:
	state = env.reset()
	done = False
	steps_whitout_reward = 0
	score = 0
	while not done:

		env.render()
		time.sleep(0.05)


		q_values = model.predict(np.reshape(state,(1,input_dim)))
		
		action = np.argmax(q_values)
		
		state, reward, done = env.step(action+1)
		

		if reward == const.WIN_REWARD:
			score+=1
			steps_whitout_reward = 0
		else:
			steps_whitout_reward+=1
			if steps_whitout_reward == MAX_STEPS_WHITOUT_REWARD:
				done = True
					
	print(score)