# Channel Estimation using Deep Learning
Course Project - Deep Learning 
Here we use the dataset ChanEst - https://data.mendeley.com/datasets/8w3zjj8hzc/1.
The goal of the first experiment is to implement two models to learn channel estimation on this dataset.
1. MLP - Multi-layer Perceptron
2. CNN - Convolutional Neural Network
Our goal is to perform experiments to compare their performance for channel estimation by varying hyperparameters and computing a few selected metrics.
## How to use this repository 
Install the dataset ChanEst in HDF5 (.h5) format. 
### Structure
CNN module - cnn.py
MLP module - multiLayerPerceptron.py
Data processing module - data_processing.py
main.py
### Requirements 
Install the requirements as follows:
```bash
pip install -r requirements.txt
``` 
 Or use conda instead of pip if you are using a conda environment. If any packages are not installed correctly they can be directly installed as:

 ```bash
 pip install <package name>
 ```
### Data Processing module
It has functions defined for extracting data from the .h5 file and a function for splitting data into train and test sets.
### CNN and MLP module
Each of them have functions defined for model design, training and evaluation.
For evaluation only MSE, NMSE and Cross-Entropy loss have been implemented yet.
### Running the main method
1. Before running the main method re-assign the file paths according to your PC. 
2. Keep in mind that the device selection specifies MPS backend which is for apple silicon. If you have a Windows/Linux PC please change it to CUDA like this

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
else:
    device = torch.device("cpu")
    print("Using CPU, CUDA not available")
``` 

Then run the code:
```bash
python main.py
```
## Results observed:
CNN outperforms MLP in terms of NMSE and Cross-Entropy loss. Further in the project we aim to sophesticate the experiment by controlling and varying certain hyperparameters.
Hyperparameters can be set and directly tuned in the main itself.