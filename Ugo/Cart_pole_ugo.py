import gym
import matplotlib.pyplot as plt
from gym.wrappers.record_video import RecordVideo
from CircularBuffer import CircularBuffer 
from model import NN
import torch as torch
import random as rd
import numpy as np

# Env creation
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.action_space.seed(42)

# Recording wrapper
wrapped_env = RecordVideo(env, video_folder = './video', episode_trigger= lambda x: x%100 == 0)
observation, info = wrapped_env.reset(seed=42)

# Rewards storage
recompenses = []
recompense = 0

# Buffer initialisation
bufferSize = 3
circularBuffer = CircularBuffer(bufferSize)
previousObservation = observation

# Model initialisation
state_space_dimension = 4
action_space_dimension = 2
policy_model = NN(state_space_dimension, action_space_dimension)
target_model = NN(state_space_dimension, action_space_dimension)
alpha = 0.01
N_target_update = 1000
use_alpha = True

if(not(use_alpha)): alpha = 0

# Stratégie epsilon gready avec decay
epsilon  = 0.99
beta = 0.999
gamma = 0.99
n_sample = 32

# Training parameters
n_train = 5000
learning_rate = 0.005
optimizer = torch.optim.SGD(policy_model.parameters(), lr=learning_rate)

for train_step in range(n_train):
    # Step
    p = rd.random()

    # Expérience
    if(p > epsilon):
        model_output = policy_model(torch.tensor(observation)).detach()
        action = torch.argmax(model_output)

        # Apprentissage
            # Récupération d'expériences passées aléatoires
        experiences = circularBuffer.sample(n_sample)
        previousObservations, observations, actions, terminateds, rewards = zip(*experiences)
        terminateds = torch.tensor(np.array(terminateds))
        rewards = torch.tensor(np.array(rewards))
        previousObservations = torch.tensor(np.array(previousObservations))
        observations = torch.tensor(np.array(observations))
        actions = torch.tensor(actions)

            # Calcul des prochaines actions avec le policy model
        next_action_policy = torch.max(policy_model(observations), dim=1)[1]

            # Calcul des Q-values avec le target model
        Q_next_observations = target_model(observations)
        Q_next_observations_max = torch.max(Q_next_observations, dim=1)[0].detach()

            # Prédiction des Q-value par le policy model
        Q_previous_observations = policy_model(previousObservations)
        Q_pred_from_previous = torch.gather(Q_previous_observations, 1, next_action_policy.view(-1, 1)).squeeze()

            # Calcul de la cible et de la loss
        Q_cible = gamma * Q_next_observations_max * (1 - terminateds) + rewards
        loss = torch.mean(torch.pow(Q_pred_from_previous - Q_cible, 2))

            # Update du policy model
        print(f"loss: {loss}, cumulated reward: {recompense}, epsilon: {epsilon}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for p in policy_model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

            # Update du target model
        if(use_alpha or train_step % N_target_update == 0):
            for i in range(len(policy_model.layers)):
                target_model.layers[i][0].weight = torch.nn.parameter.Parameter((1 - alpha) *  policy_model.layers[i][0].weight + alpha * target_model.layers[i][0].weight)

        action = action.item()

    # Exploration
    else:
        action = wrapped_env.action_space.sample()
    
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    recompense += reward

    epsilon = epsilon * beta

    # Store action
    if(train_step>0):
        circularBuffer.push((previousObservation, observation, action,int(terminated), reward))
    previousObservation = observation

    # End episode
    if terminated or truncated:
        recompenses.append(recompense)
        observation, info = wrapped_env.reset()
        recompense = 0

# End experience
wrapped_env.close()
plt.scatter(list(range(len(recompenses))), recompenses)
plt.ylabel("rewards")
plt.xlabel("episode")
plt.show()