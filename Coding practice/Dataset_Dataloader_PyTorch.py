import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#create a class for the dataset, in this case I used wine.csv
class WineDataset(Dataset):

#initialization
    def __init__(self):
        #load the csv file
        xy = np.loadtxt('D:/User/Sem Aug 2021/SS-4290/PyTorch/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        #slice the samples starting with the first column
        #convert numpy to tensors
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        #get the number of samples
        self.n_samples = xy.shape[0]

#get item function
    def __getitem__(self, item):
        #will return a tuple
        return self.x[item], self.y[item]
#return total no of samples
    def __len__(self):
        return self.n_samples

#constructor
dataset = WineDataset()
#get the first row vector
#first_data = dataset[0]
#features, labels = first_data
#print("First row vector: ")
#print(features, labels)

#Using dataloader
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

#Perform iteration
datatiter = iter(dataloader)
data = datatiter.next()
features, labels = data
print("Iteration for 4 times: ")
print(features, labels)

#training loop
#initialiaze epoch to 2
num_epochs = 2
#get total no of samples
total_samples = len(dataset)
#get iteration by dividing total samples with 4(batch_size)
n_iterations = math.ceil(total_samples/4)
#print the total sample and number of iteration per epoch
print("total samples and iteration: ")
print(total_samples, n_iterations)
print(" ")

#for loop in the range of no of epoch
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #get info every 5 steps
        if(i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
