"""
Experiment 1: Main Method - Projet Apprentissage Profond
Author: Harshit Kale
Created on: Mer. 4 Nov 2026 17:58

"""

from multiLayerPerceptron import mlp_model, train_mlp, test_mlp
from data_processing import dataProcessor
import matplotlib.pyplot as plt
from cnn import cnn_model, train_cnn, test_cnn
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available")
else:
    device = torch.device("cpu")
    print("Using CPU, MPS not available")

""" Data extraction & loading """

# Data extraction object
dp = dataProcessor("/Users/harshitkale/Documents/Apprentissage Profond/Channel Estimation using Deep Learning/6G_ChanEst_Dataset_10k_Samples.h5")
X, Y = dp.extractData()

print(X.shape)
print(Y.shape)
# [K, L, C, N]
# Shape = (10000, 2, 14, 612) => OFDM 612 subcarriers, 14 symbols, 10000 samples, 2 packed channels
batch_size = 1000
# Slpit datasets:
train_dataloader, test_dataloader = dp.splitData(X, Y, batch_size)

""" Initialize Model """
# MLP Model parameters:
input_size = X[0].numel()
output_size = Y[0].numel()
hidden_size = 512  # Initial experimental value

# MLP model object instance
mlp_estimator = mlp_model(input_size, hidden_size, output_size).to(device)

# CNN model parameters hardcoded in method

cnn_estimator = cnn_model().to(device)

""" Training the model """
# Training parameters
loss = "mse_loss"
optimizer = "Adam"
num_epochs = 10
lr = 0.001

# training object instance
mlp_trainer = train_mlp(train_dataloader, mlp_estimator, loss, optimizer, num_epochs, lr)
cnn_trainer = train_cnn(train_dataloader, cnn_estimator, loss, optimizer, num_epochs, lr)

# train the model
epochs_mlp, loss_mlp = mlp_trainer.train(device)
epochs_cnn, loss_cnn = cnn_trainer.train(device)

""" Testing the model """

mlp_evaluator = test_mlp(test_dataloader, mlp_estimator)
mlp_accuracy = mlp_evaluator.evaluate(device)
mlp_nmse = mlp_evaluator.nmse(device)

cnn_evaluator = test_cnn(test_dataloader, cnn_estimator)
cnn_accuracy = cnn_evaluator.evaluate(device)
cnn_nmse = cnn_evaluator.nmse(device)

print("MLP:")
print(f"Test MSE (Mean square error) is {mlp_accuracy:.2f}")
print(f"NMSE in dB is {mlp_nmse:.3f}")
print("CNN:")
print(f"Test MSE (Mean square error) is {cnn_accuracy:.2f}")
print(f"NMSE in dB is {cnn_nmse:.3f}")


""" Visualize """
plt.plot(epochs_mlp, loss_mlp, label='MLP')
plt.plot(epochs_cnn, loss_cnn, label='CNN')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title("Loss Diagram for MLP and CNN")
plt.show()



