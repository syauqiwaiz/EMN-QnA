import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

#Training and testing datasets downloaded from The Internet
#Two sets of tensors: One containing the image; the other the label
train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

#Train and test the data with batch size of 10
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#Display the train dataset:
for data in trainset:
    print(data)
    break

#Assign the first image in the tensor to a variable
x, y = data[0][0], data[1][0]
print(y)

#Display the dimension of the image
print(data[0][0].shape)

#Display the image
plt.imshow(data[0][0].view(28,28))
plt.show()

