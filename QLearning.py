# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:02:57 2021

@author: gavh_
"""


import numpy as np
import gym
import matplotlib.pyplot as plt

# Inicializamos el ambiente de MountainCar de OpenAI Gym
env = gym.make('MountainCar-v0') #Seleccionamos el ambiente MountainCar-v0
env.reset() #Reiniciamos el ambiente al principio (y obtenemos el estado inicial)

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):

  '''Obtener el número de estados posibles para nuetro problema'''
  #En este caso, hay estados infinitos. En esta implementación, discretizaremos estos valores para tener 10 posiciones y 100 velocidades
  num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
  num_states = np.round(num_states, 0).astype(int) + 1

  '''Inicializamos la Tabla Q'''
  #Inicialmente, la Tabla Q tiene valores aleatorios en el rango [-1, 1]. Hay que notar que la Tabla Q tiene tres dimensiones, pues cada valor
  #corresponde a una posicion, a una velocidad, y a una acción
  Q = np.random.uniform(low = -1, high = 1, 
                        size = (num_states[0], num_states[1], env.action_space.n))

  '''Listas para almacenar las recompensas'''
  reward_list = []
  ave_reward_list = []

  '''Reducción de epsilon'''
  #En cada episodio, epsilon se reducirá ligeramente, de manera que al principio, el algoritmo explore nuevas alternativas de manera aleatoria, 
  #mientras que en los episodios finales, el algoritmo será más conservador y elegirá acciones ya visitadas
  reduction = (epsilon - min_eps)/episodes

  '''Inicio del Aprendizaje Q'''
  for i in range(episodes): #Iterando sobre el número de episodios
    
    '''Inicializamos parametros'''
    done = False #Variable para dar seguimiento al progreso
    tot_reward, reward = 0,0 #Recompensa inicial
    state = env.reset() #Estado inicial (reiniciar ambiente en cada episodio)

    '''Seleccionar el estado inicial discretizado'''
    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)

    while done != True: #Mientras no haya exito
      
      '''Renderizar el ambiente gráfico (Cada 20 episodios)'''
      if i >= (episodes - 20) or i == 5:
        env.render()

      '''Elegir nuestra siguiente acción'''
      #Utilizamos epsilon como variable aleatoria
      if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state_adj[0], state_adj[1]]) 
      else:
        action = np.random.randint(0, env.action_space.n)
      
      '''Obtener el siguiente estado y la recompensa asociada'''
      state2, reward, done, info = env.step(action) 

      '''Obtenemos la versión discreta del nuevo estado'''
      state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
      state2_adj = np.round(state2_adj, 0).astype(int)

      '''Agregar la recompensa a Q si el coche llega a la meta'''
      #En caso contrario, ajustamos la Tabla Q con la ecuación vista en la presentación
      if done and state2[0] >= 0.5:
        Q[state_adj[0], state_adj[1], action] = reward
      
      else:
        delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])
        Q[state_adj[0], state_adj[1],action] += delta #Ajustamos el nuevo valor
      
    '''Actualizamos variables'''
    tot_reward += reward #Actualizar recompensa total
    state_adj = state2_adj #Actualizar nuevo estado

    '''Reducción de epsilon'''
    if epsilon > min_eps:
      epsilon -= reduction
  
    '''Seguimiento a la recompensa total'''
    reward_list.append(tot_reward)

    if (i+1) % 100 == 0:
      ave_reward = np.mean(reward_list) #Promediar recompensas
      ave_reward_list.append(ave_reward)
      reward_list = []
    
    if (i+1) % 100 == 0:    
            print('Episodio {}, Recompensa promedio: {}'.format(i+1, ave_reward))
    
  '''Al final, cerramos el ambiente'''
  env.close()

  return ave_reward_list #Devolvemos la lista de recompensas promedio

'''Llamamos al algoritmo'''
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 5000)

# Graficas
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensa promedio')
plt.title('Evolución de la recompensa promedio')
    

  
