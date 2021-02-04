'''

En este archivo se entrenará un modelo de redes neuronales con DQN.
Este tipo de redes reciben como input el estado actual del environment
y devuelven una serie de Q-Values, cada Q-Value se corresponde a una
acción posible en el environment y representa qué tan buena es dicha acción
para ese estado.
A medida que el sistema vaya jugando diferentes partidas, va a ir guardando
experiencias pasadas en un buffer llamado 'replay_memory', con el cual va a
poder entrenar la red. De este buffer, se tomarán acciones aleatorias con el
fin de no condicionar el entrenamiento de la red, es decir, en un environment
los estados contiguos tienen mucha similitud, por lo que si a una red la
entreno con todos estados contiguos, mi set de entrenamiento no resulta muy
robusto. Por esto mismo, se guarda en un buffer de tamaño considerado las
experiencias pasadas con el fin de tomar aleatoriamente muestras para el 
entrenamiento. Para ajustar los Q-Values óptimos se utiliza la ecuación
de Bellman.
Además, para un entrenamiento más eficiente se cuenta con 2 redes neuronales
que poseen igual arquitectura, pero distintos pesos, estas son main_model y
target_model. El entrenamiento principal se encuentra en el main_model, y el
target_model irá copiando los pesos del main_model cada cierto numero de
iteraciones. El target_model será el encargado de predecir los Q-values para
acciones futuras mientras que el main_model predecirá las acciones para el
estado actual.



'''




from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD

import numpy as np
import time
from collections import deque
import random

import constants as const
from snake_env import SnakeEnv
from snake import Snake

# Episodes
train_episodes = 15_000

# input_dim = dimensionalidad del estado del environment, en este caso 20
input_dim = 20
# output_dim = acciones posibles a realizar en el environment
output_dim = 4


env = SnakeEnv(const.SCREEN_SIZE)


'''
La capa de salida tiene tantas neuronas como acciones posibles y tiene como
funcion de activacion 'linear', esto es porque la salida representa un Q-value
'''
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
    '''
    Coeficiente de exploración
    Con este epsilon, se obtendrá un comportamiento más aleatorio en las
    primeras simulaciones.
    Epsilon comenzará en 1 y decaerá exponencialmente hasta 0.01 (nunca llega
    a ser 0, esto le da la capacidad de descubrir nuevos estados en cualquier
    momento del entrenamiento)
    '''
    epsilon = 1
    max_epsilon = 1 
    min_epsilon = 0.01
    decay = 0.01

    # Inicializar modelos
    model = create_model(input_dim,output_dim)
    target_model = create_model(input_dim,output_dim)
    target_model.set_weights(model.get_weights())

    # Buffer de experiencias pasadas, guarda las últimas 50000 experiencias
    replay_memory = deque(maxlen=50_000)

    # Contador para copiar los valores del model al target_model
    steps_to_update_target_model = 0

    max_score = 0

    for episode in range(train_episodes):
        
        observation = env.reset()
        done = False

        # Algunas métricas para ver durante el entrenamiento
        total_training_rewards = 0
        score = 0

        # Cuenta los pasos sin obtener recompenza para saber si el agente 
        # está en un bucle infinito
        steps_whitout_reward = 0

        while not done:
            steps_to_update_target_model += 1
            
            # Si se desea entrenar el modelo en alguna plataforma como Colab,
            # se debe poner False en este if o comentar las siguientes líneas
            if True:
                env.render()
                time.sleep(0.1)


            # Exploración
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = np.random.randint(4)
            else:
                #Preprocesamiento del estado
                encoded_reshaped = np.array(observation).reshape((1,)+ np.array(observation).shape)
                #Predicción del modelo
                predicted = model.predict(encoded_reshaped).flatten()
                #Acción a realizar
                action = np.argmax(predicted)

            # Ejecuta la acción
            new_observation, reward, done = env.step(action+1)

            # Si se obtuvo recompensa no estamos en bucle infinito y aumentamos
            # el score, sino, verifico que no esté en bucle infinito.
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

            # Agrego el estado del sistema al buffer de experiencias.
            replay_memory.append([observation, action, reward, new_observation, done])

            # Entreno el modelo utilizando la ecuación de Bellman
            if steps_to_update_target_model % 10 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            # Actualizo el nuevo estado
            observation = new_observation

            # Actualizo métrica
            total_training_rewards += reward

            if done:
                # Análisis de la simulación
                print('Epoca: {} Score: {} MaxScore: {} AcumulatedReward: {}'.format(episode, score, max_score, total_training_rewards))
                total_training_rewards += 1

                # Actualizo target_model
                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0

        # Al finalizar un episodio, actualizo la variable de exploración
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

    print('Saving model...')
    model.save('models/model.h5')
                    


if __name__ == '__main__':
    main()
