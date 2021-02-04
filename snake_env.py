'''
El environment es un entorno de simulación que ejecuta acciones en función de
un input recibido, y para cada acción actualiza su estado y devuelve un reward
Para hacer esto posible, el environment debe tener 3 funcionalidades principales:
	-Reset: inicializa el sistema y devuelve el estado inicial
	-Step: actualiza el estado en función de la accion recibida y devuelve el
	nuevo estado y una recompenza por la acción realizada (esta puede ser
	positiva o negativa)
	-Render: muestra de forma gráfica el estado del sistema

Dicho esto, este environment simula el juego Snake, donde el jugador debe ir
agarrando las comidas que aparecen de manera aleatoria y el cuerpo de la 
serpiente irá creciendo. De la misma manera, el jugador puede morir chocando 
contra una pared (fin de la pantalla) o contra su cuerpo. El jugador recibe 
una recompenza de 5 en caso de alcanzar una comida y una recompenza de -5 en 
caso de morir, en caso contrario la recompenza es 0

La codificación del estado consta de 20 valores. Los 8 primeros representan
la distancia a un obstáculo (sea pared o cuerpo) en 8 direcciones. Las
siguientes 4 representan la distancia a la comida en 4 direcciones. Luego se
codifica en one-hot la direccion de la cabeza de la serpiente y la dirección
de la cola, formando así los últimos 8 valores
'''


from PIL import Image
import cv2
import numpy as np
from snake import Snake, Food
import constants as const
import random
from collections import deque

class SnakeEnv:
	def __init__(self,size):
		self.size = size
		self.snake = Snake(size)
		self.food = Food(size)
		self.key = self.snake.direction

	def reset(self):
		del self.snake
		del self.food
		self.snake = Snake(self.size)
		self.food = Food(self.size)
		self.key = self.snake.direction
		state = self.calculate_state()
		return state

	def step(self,key=-1):
		if key > 0:
			self.key = key
		live = self.snake.step(self.key)

		state = self.calculate_state()
		done = False
		reward = 0

		if live == False:
			done = True
			reward = const.LOSS_REWARD
		if(self.snake.x == self.food.x and self.snake.y == self.food.y):
			empty_pos = self.find_empty_pos()
			self.food.regenerate(empty_pos)
			self.snake.got_food = True
			reward = const.WIN_REWARD

		

		return state, reward, done

	def render(self):
		img = self.get_image()
		img = cv2.resize(np.array(img), (const.SCREEN_PLOT_SIZE,const.SCREEN_PLOT_SIZE), interpolation = cv2.INTER_AREA)
		cv2.imshow("image", np.array(img))
		cv2.waitKey(1)
		#if save_path != None:
			#pepe = cv2.imwrite(save_path, img) 
			#print('pepe')

	
	def get_image(self):
		env = np.zeros((self.size, self.size, 3), dtype=np.uint8) #BACKGROUND

		env[self.snake.y,self.snake.x] = const.SNAKE_HEAD_COLOR #SNAKE HEAD

		for body in self.snake.body: #SNAKE BODY
			env[body.y,body.x] = const.SNAKE_BODY_COLOR

		env[self.food.y,self.food.x] = const.FOOD_COLOR

		img = Image.fromarray(env, 'RGB')
		return img

	def calculate_state(self):
        #State:

        #distance_to_obstacle_top [1]
        #distance_to_obstacle_right [1]
        #distance_to_obstacle_bottom [1]
        #distance_to_obstacle_left [1]
        #distance_to_obstacle_top-right [1]
        #distance_to_obstacle_right-bottom [1]
        #distance_to_obstacle_bottom-left [1]
        #distance_to_obstacle_left-top [1]

        #distance_to_food_top [1]
        #distance_to_food_right [1]
        #distance_to_food_bottom [1]
        #distance_to_food_left [1]

        #is_direction_top? [1]
        #is_direction_right? [1]
        #is_direction_bottom? [1]
        #is_direction_left? [1]

        #is_tail_direction_top? [1]
        #is_tail_direction_right? [1]
        #is_tail_direction_bottom? [1]
        #is_tail_direction_left? [1]
		
		
		distance_to_obstacle = self.calculate_obstacle_distance()
		food_directions = self.calculate_food_direction()
		snake_direction = [0,0,0,0]
		snake_direction[self.snake.direction-1] = 1
		tail_direction = [0,0,0,0]
		tail_direction[self.calculate_tail_direction()] = 1

		state = np.array([distance_to_obstacle[0]/self.size,
						  distance_to_obstacle[1]/self.size,
						  distance_to_obstacle[2]/self.size,
						  distance_to_obstacle[3]/self.size,
						  distance_to_obstacle[4]/self.size,
						  distance_to_obstacle[5]/self.size,
						  distance_to_obstacle[6]/self.size,
						  distance_to_obstacle[7]/self.size,
						  food_directions[0],
						  food_directions[1], 
						  food_directions[2], 
						  food_directions[3],
						  snake_direction[0],
						  snake_direction[1],
						  snake_direction[2],
						  snake_direction[3],
						  tail_direction[0],
						  tail_direction[1],
						  tail_direction[2],
						  tail_direction[3]])

		return state
		
	def calculate_food_direction(self):

		food_directions = np.zeros(4, dtype=int)
		dist = np.zeros(2)
		dist[0] = self.food.x - self.snake.x
		dist[1] = self.food.y - self.snake.y
		#dist = np.array([self.food.x,self.food.y]) - np.array([self.snake.x,self.snake.y])

		if dist[0] > 0:
			food_directions[1] = 1 # right
		elif dist[0] < 0:
			food_directions[3] = 1 # left

		if dist[1] > 0:
			food_directions[0] = 1 # up
		elif dist[1] < 0:
			food_directions[2] = 1 # left

		return food_directions

	def find_empty_pos(self):
		empty_pos = []
		is_empty = True

		for posx in range(const.SCREEN_SIZE):
			for posy in range(const.SCREEN_SIZE):
				if not (self.snake.x == posx and self.snake.y == posy):
					for b in self.snake.body:
						if (b.x == posx and b.y == posy):
							is_empty = False
					if is_empty:
						empty_pos.append([posx,posy])
					is_empty = True

		random_empty_pos = random.choice(empty_pos)
		return random_empty_pos


	def calculate_obstacle_distance(self):
		sy = self.snake.y
		sx = self.snake.x
		max_distance =  [sy, #top
					     self.size - sx, #right
					     self.size - sy, #bottom
					     sx, #left
					     min(sy,self.size-sx), #top-right
						 min(self.size-sx,self.size-sy), #right-bottom
						 min(self.size-sy, sx), #bottom-left
						 min(sx,sy)] #left-top

		for b in self.snake.body:
			if b.y == sy:
				if 0 < b.x - sx < max_distance[1]:
					max_distance[1] = b.x - sx
				if b.x < max_distance[3]:
					max_distance[3] = sx - b.x
			if b.x == sx:
				if 0 < b.y - sy < max_distance[2]:
					max_distance[2] = b.y - sy
				if b.y < max_distance[0]:
					max_distance[0] = sy - b.y
			if abs(b.y - sy) == abs(b.x - sx):
				if b.y - sy > 0:
					if b.x - sx > 0:
						if max_distance[5] < b.x - sx:
							max_distance[5] = abs(b.x - sx)
					else:
						if max_distance[6] < sx - b.x:
							max_distance[6] = abs(b.x - sx)
				else:
					if b.x - sx > 0:
						if max_distance[4] < b.x - sx:
							max_distance[4] = abs(b.x - sx)
					else:
						if max_distance[7] < sx - b.x:
							max_distance[7] = abs(b.x - sx)


		return max_distance


	def calculate_tail_direction(self):
		tx = self.snake.body[-1].x
		ty = self.snake.body[-1].y
		if len(self.snake.body) > 1:
			bx = self.snake.body[-2].x
			by = self.snake.body[-2].y
		else:
			bx = self.snake.x
			by = self.snake.y

		if bx-tx == 0:
			if by-ty > 0:
				direction = 2
			else:
				direction = 0
		else:
			if bx-tx > 0:
				direction = 1
			else:
				direction = 3

		return direction


