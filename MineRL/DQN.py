import torch
from torch import nn

#Création de notre DQN
class DQN(torch.nn.Module):
    
    #Détail de notre DQN
    def __init__(self, input_size, output_size): 
        super(DQN, self).__init__() 
        input_width, input_height, input_channels = input_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 10, (7, 7), stride = 7),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 5, (2, 2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(5, 1, (2, 2), stride=2, padding = 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(25, 16),
            nn.ReLU())
            
        self.layer5 = nn.Sequential(
            nn.Linear(16, output_size),
            nn.ReLU())

    #Utilisation du DQN
    def forward(self, x): 
        if(x.dim()==3):
            x = x[None, :]
        x = torch.swapaxes(x, 1, 3)
        x = torch.swapaxes(x, 2, 3)
        x1 = self.layer1(x.float())
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        return x5
    
    #Entraînement sur un batch de données
    def train_on_batch(self, target_model, optimizer, obs, acts, rewards, next_obs, 
                       terminals, gamma=0.99):
        
        #Calcul des next_act avec le model de base
        next_q_values = self.forward(next_obs)
        max_next_acts = torch.max(next_q_values, dim=1)[1].detach()
        
        #Calcul des q_values avec le target model
        target_next_q_values = target_model.forward(next_obs)
        max_next_q_values = target_next_q_values.gather(index=max_next_acts.view(-1, 1), dim=1)
        max_next_q_values = max_next_q_values.view(-1).detach()        
        
        #Calcul de la "vraie" q value
        terminal_mods = 1 - terminals
        actual_qs = rewards + terminal_mods * gamma * max_next_q_values
            
        pred_qs = self.forward(obs)
        pred_qs = pred_qs.gather(index=acts, dim=1)
        pred_qs = torch.max(pred_qs, 1).values
        loss = torch.mean((actual_qs - pred_qs) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
