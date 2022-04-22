import torch

#create a tensor with gradient (requires_grad=False on default)
x = torch.randn(3, requires_grad=True)
print('requires_grad is included in tensor: ')
print(x)
print(" ")

#perform addition on tensors
y = x+2
print("grad_fn of AddBackward is included in tensor if we perform addition: ")
print(y)
print(" ")

#perform multiplication and mean on tensors
z = y*y*2
z= z.mean()
print("grad_fn of MeanBackward is included in tensor if we perform mean: ")
print(z)
print(" ")

#Find the gradient using backward() function, it is essentially dz/dx
z.backward()
print("gradient:")
print(x.grad)
print(" ")

#x.grad only accept scalar output, therefore for multiple values the operation is different
print("create a tensor of size 3 with grad:")
a = torch.randn(3, requires_grad=True)
print(a)
b = a+2
#using c as the input
c = b*b*2
#create a tensor with vector of the same size
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
#passed an argument on backward()
c.backward(v)
print("the gradient using vector outputs: ")
print(x.grad)
print(" ")

#in the case where we do want the gradient to interfere with operation
print("create a tensor of size 3 with grad: ")
d = torch.randn(3, requires_grad=True)
print(d)
#passed any of these 3 arguments
print("1. Use requires_grad_(False): ")
print(d.requires_grad_(False))
print("2. Use detach(): ")
e = d.detach()
print(e)
print("3. torch.no_grad(): ")
with torch.no_grad():
    f = d + 2
    print(f)
print(" ")

#whenever we called backward(), the gradient accumulate in .grad(), so that needs to change
print("create tensor of size 4 with grad into a for loop and see if the grad accumulates: ")
weights = torch.rand(4, requires_grad=True)
#create a for loop to observe if it accumulates
for i in range(3):
    model = (weights*3).sum()

    model.backward()

    print(weights.grad)
#use grad.zero_() so that it does not accumulate
    weights.grad.zero_()


