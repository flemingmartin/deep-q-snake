import numpy as np

class Food:
	def __init__(self,size):
		self.x = np.random.randint(0,size)
		self.y = np.random.randint(0,size)

	def regenerate(self,empty_pos):
		self.x = empty_pos[0]
		self.y = empty_pos[1]

class BodySnake:
	def __init__(self,pos):
		self.x = pos[0]
		self.y = pos[1]

class Snake:
	def __init__(self,size):
		self.map_size = size
		self.x = np.random.randint(0+2,size-2)
		self.y = np.random.randint(0+2,size-2)

		self.body_len = 1
		self.body = []
		self.direction = self.init_random_body()
		self.got_food = False

	def step(self,key):
		self.update_body()
		k = ((key+1)%4)+1
		if k != self.direction:
			self.direction = key

		newX, newY = self.new_position(self.direction)
		valid_position = self.check_valid((newX,newY))

		
		if valid_position:
			self.x = newX
			self.y = newY
			return True
		else:
			return False
	
	def new_position(self,direction):
		if direction == 1:
			return (self.x,self.y-1)
		elif direction == 2:
			return (self.x+1,self.y)
		elif direction == 3:
			return (self.x,self.y+1)
		elif direction == 4:
			return (self.x-1,self.y)

	def check_valid(self,pos):
		if pos[0] == -1 or pos[0] == self.map_size:
			return False
		if pos[1] == -1 or pos[1] == self.map_size:
			return False

		for b in self.body:
			if pos[0] == b.x and pos[1] == b.y:
				return False

		return True


	def update_body(self):
		next_x = self.x
		next_y = self.y
		for b in self.body:
			my_x = b.x
			my_y = b.y
			b.x = next_x
			b.y = next_y
			next_x = my_x
			next_y = my_y

		if self.got_food:
			self.body_len += 1
			self.body.append(BodySnake((next_x,next_y)))
			self.got_food = False



	def init_random_body(self):
		random_pos = np.random.randint(1,4)
		if random_pos == 1:
			pos_body = [self.x+1 , self.y]
			direction = 4 #LEFT
		elif random_pos == 2:
			pos_body = [self.x , self.y+1]
			direction = 1 #UP
		elif random_pos == 3:
			pos_body = [self.x-1 , self.y]
			direction = 2 #RIGHT
		elif random_pos == 4:
			pos_body = [self.x , self.y-1]
			direction = 3 #DOWN


		self.body.append(BodySnake(pos_body))
		return direction