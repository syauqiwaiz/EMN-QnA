import torch

#initialize two sets of tensors
x = torch.tensor([5,3])
y = torch.Tensor([2,1])

#multiple the two arrays
print("Multiplication of the two arrays: ")
print(x*y)

#create an tensors of 5 elements in 2 arrays:
x = torch.zeros([2,5])
print("New tensor: ")
print(x)

#check the size of tensors
print("Tensor size: ")
print(x.shape)

#Assign random elements in tensor
y = torch.rand([2,5])
print("Random elements assigned:")
print(y)

#View the different size of the tensors:
print("Viewing different Tensor's size ([1, 10]: ")
print(y.view([1,10]))

print("Print y again and it returns back to original size: ")
print(y)

#Change the size of tensor
y = y.view([1,10])
print("Change the size of tensor: ")
print(y)
