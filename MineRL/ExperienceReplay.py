import numpy as np
import torch
import random as rd

#Création de notre experience replay
class ExperienceReplay:
    
    #Initialise un ER de taille capacity
    def __init__(self,capacity):
        self.capacity = capacity
        self.data = []
    
    #Ajouter les données d'un step à notre ER
    def add_step(self,step_data):
        self.data.append(step_data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]
    
    #Retourne un échantillon de notre ER
    def sample(self, n):
        n = min(n,len(self.data))
        indices = np.random.choice(range(len(self.data)), n, replace=False)
        samples = np.asarray(self.data)[indices]
        
        state_data = torch.tensor(np.stack(samples[:, 0])).float()
        act_data = torch.tensor(np.stack(samples[:, 1])).long()
        reward_data = torch.tensor(np.stack(samples[:, 2])).float()
        next_state_data = torch.tensor(np.stack(samples[:, 3])).float()
        terminal_data = torch.tensor(np.stack(samples[:, 4])).float()
        
        return state_data, act_data, reward_data, next_state_data, terminal_data
