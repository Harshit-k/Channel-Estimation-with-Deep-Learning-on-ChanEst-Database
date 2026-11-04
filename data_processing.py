# Data Preprocessing Module
import h5py
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

class dataProcessor():
    def __init__(self, data_path):
        super(dataProcessor, self).__init__()
        self.data_path = data_path

    def extractData(self):
        
        with h5py.File(self.data_path, 'r') as f:
            # 1. Inspect the file structure
            print(f.keys())         

            # 2. Read tensors as NumPy arrays
            X = f['X_input'][()]    # Input
            Y = f['Y_label'][()]    # Output
            
            # 3. Check metadata / attributes
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    print(f"{key}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{key}: group -> {list(obj.keys())}")
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    def splitData(self, X, Y, batch_size):
        dataset = TensorDataset(X, Y)
        # 2. Define split sizes (e.g., 80% train, 20% test)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # 3. Perform the split
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

        return train_dataloader, test_dataloader
  