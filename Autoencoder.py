import torch
import torch.nn as nn
import torch.optim as optim
import os

class Autoencoder(nn.Module):
    def __init__(self,input_dim=1024,latent_dim=20):
        super(Autoencoder,self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.file=f'./autoencoder_weights_input_{self.input_dim}_out_{self.latent_dim}.pth'

        self.encoder= nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), int(input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(input_dim/4), int(latent_dim))
        )
        self.decoder= nn.Sequential(
            nn.Linear(int(latent_dim), int(input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(input_dim/4), int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), int(input_dim))
        )

        
    def forward(self,x):
        x = self.encoder(x)
        x=self.decoder(x)
        return x
    
    def training_phase(self, features,device, num_epochs=50, learning_rate=1e-3):
        self.to(device)  # Sposta il modello sul dispositivo specificato
        if os.path.exists(self.file):
            self.load_state_dict(torch.load(self.file))
        else:
            criterion=nn.MSELoss()
            optimizer=optim.Adam(self.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                for i in range(features.shape[0]):
                    inputs = features[i].to(device)
                    outputs = self.forward(inputs)
                    
                    loss=criterion(outputs, inputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    del inputs, outputs
                    
                print(f'epoch {epoch} training loss:{loss}')
            self.store_weights()
    
    def store_weights(self):
        torch.save(self.state_dict(), self.file)