import gym
import minerl
import logging
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import environment
from DQN import DQN
from ExperienceReplay import ExperienceReplay
import torch
import copy
import numpy as np


# logging.basicConfig(level=logging.DEBUG)
#Initialisation de l'environnement
env = gym.make('Mineline-v0')
video_recorder = VideoRecorder(env, path = './video.mp4')
observation = env.reset()
# print(observation['pov'])

#Initialisation des différents paramètres
n_episodes = 1500    #nbr épisodes max pour training
max_steps = 500    #nbr step max par épisode
er_capacity = 2000    #capacité de notre ER
input_size = env.observation_space['pov'].shape   #taille de l'input de notre réseau

action_keys = ["attack", "left", "right"]
output_size = len(action_keys)    #taille de sortie
update_freq = 1    #fréquence de mise à jour de notre réseau
target_update_delay = 100    #fréquence de mise à jour du target_model
n_anneal_steps = 5000    #paramètre pour contrôler l'exploration
epsilon = lambda step: np.clip(1 - 0.9 * (step/n_anneal_steps), 0.25, 1)   #fonction permettant de contrôler lexploration
train_batch_size = 32    #taille d'un batch d'entraînement
learning_rate = 0.00025
print_freq = 20

#Initialisation de l'ER, du model, du target_model
er = ExperienceReplay(er_capacity)
model = DQN(input_size, output_size)
target_model = copy.deepcopy(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
all_rewards = []
global_step = 0

for episode in range(n_episodes):
    
    #Reset de l'environnement et des variables pour chaque nouvel épisode
    obs = env.reset()
    obs = torch.tensor(obs["pov"].copy())
    
    episode_reward = 0
    step = 0
    
    while step < max_steps:

        #Début du step
        #Choix d'une action random pour explorer ou non en fonction de la valeur d'epsilon
        if np.random.rand() < epsilon(global_step):
            act = env.action_space.sample()
        else:
            q_values = model(obs)
            q_values = q_values.cpu().detach().numpy()
            act_number = np.argmax(q_values)
            act = dict.fromkeys(action_keys, 0)
            act[action_keys[act_number]]=1

        act_array = [act[action_key] for action_key in action_keys]
        
        #Récuperation des données suite à l'action faite
        next_obs, reward, terminated, truncated = env.step(act)
        next_obs = torch.tensor(next_obs["pov"].copy())
        episode_reward += reward
        
        #Ajout des données à l'ER
        er.add_step([obs, act_array, reward, next_obs, int(terminated)])
        obs = next_obs
        
        #Train sur un batch
        if global_step % update_freq == 0:
            obs_data, act_data, reward_data, next_obs_data, terminal_data = er.sample(train_batch_size)
            model.train_on_batch(target_model, optimizer, obs_data, act_data,
                                 reward_data, next_obs_data, terminal_data)
        
        #Mise à jour du target_model
        if global_step and global_step % target_update_delay == 0:
            target_model = copy.deepcopy(model)
            

        #Fin du step
        step += 1
        global_step += 1
        
        if terminated:
            break

    all_rewards.append(episode_reward)
    
    if episode % print_freq == 0:
        print('Episode #{} | Step #{} | Epsilon {:.2f} | Avg. Reward {:.2f}'.format(
            episode, global_step, epsilon(global_step), np.mean(all_rewards[-print_freq:])))
    
    #Fin de l'apprentissage si on a une moyenne satisfaisante sur les derniers essais + sauvegarde du modèle
    if np.mean(all_rewards[-print_freq:]) >= 470:
        print(episode)
        save = model.state_dict()
        break
        
env.close()
video_recorder.close()