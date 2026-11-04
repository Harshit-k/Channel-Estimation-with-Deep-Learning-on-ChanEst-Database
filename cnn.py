# Convolutional Neural Network Module

import torch
import torch.nn as nn
import numpy as np

class cnn_model(nn.Module):

    def __init__(self):
        super(cnn_model, self).__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 2, kernel_size=3, padding=1)

        )

    def forward(self, x):
        return self.layers(x)

class train_cnn():
    def __init__(self, train_dataloader, model, loss, optimizer, num_epochs, lr):
        super(train_cnn, self).__init__()

        self.train_dataloader = train_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr = lr

    def getLoss(self, type):
        if type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif type == "mse_loss":
            return nn.MSELoss()
        elif type == "L1":
            return nn.L1Loss()
        else:
            print("Invalid argument for loss type")
        
    def getOptimizer(self, type):
        if type == "Adam":
            return torch.optim.Adam(self.model.parameters(), self.lr)
        elif type == "sgd":
            return torch.optim.SGD(self.model.parameters(), self.lr)
        else:
            print("Invalid argument for optimizer type")


    def train(self, device):

        criterion = self.getLoss(self.loss)
        optimizer = self.getOptimizer(self.optimizer)

        epochs = []
        loss = []
        # Training loop:
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, Y_batch in self.train_dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                reward = criterion(outputs, Y_batch)
                reward.backward()
                optimizer.step()
                running_loss += reward.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_dataloader):.4f}")
            epochs.append(epoch)
            loss.append(running_loss / len(self.train_dataloader))

        return epochs, loss
    
class test_cnn():
    def __init__(self, test_dataloader, model):
        super(test_cnn, self).__init__()
        self.test_dataloader = test_dataloader
        self.model = model

    def evaluate(self, device):

        self.model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for X_batch, Y_batch in self.test_dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                outputs = self.model(X_batch)

                total_loss += criterion(outputs, Y_batch).item()

        average_loss = total_loss / len(self.test_dataloader)

        return average_loss
    
    def nmse(self, device):
        nmse = 0.0
        numerator = 0.0
        denominator = 0.0

        with torch.no_grad():
            for X_batch, Y_batch in self.test_dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                outputs = self.model(X_batch)
                numerator += torch.sum((outputs - Y_batch)**2).item()
                denominator += torch.sum(Y_batch**2).item()

        nmse = numerator / denominator
        nmse_db = 10 * np.log10(nmse)
        
        return nmse_db






        