import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()

        # self.act = nn.ReLU()
        # self.linearLayer = nn.Linear(input_size, output_size)
        # self.layers = [self.linearLayer]

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(64, output_size))

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
    def forward(self, input):
        # linearOutput = self.linearLayer(input)
        # return self.act(linearOutput)
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3)
        return x4
