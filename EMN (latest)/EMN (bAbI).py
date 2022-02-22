import torch
import torch.nn as nn
from torchtext.legacy.datasets import BABI20


#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK, device=torch.device("cpu"))
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab

# Use dataloader to split train, test, valid and vocab
train_iter, valid_iter, test_iter, vocab = dataloader(batch_size=32, memory_size=50, task=1, joint=False, tenK=False)

# Parameters
V = len(vocab)
d = 3

class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create Embeddings
        self.embedA = nn.Embedding(V, d)
        self.embedB = nn.Embedding(V, d)
        self.embedC = nn.Embedding(V, d)
        self.LinW = nn.Linear(d, V)

    def forward(self, story, query):

        ext_memory = []

        for sentence in story:
            mi = self.embedA(sentence) #mi: torch.Size([50, 6, 3])
            #print("mi: ", mi.shape)
            ext_memory.append(mi)

        a_embed = torch.sum(torch.stack(ext_memory), dim=1) #a_embed: torch.Size([32, 6, 3])
        #print("a_embed: ", a_embed.shape)

        u = self.embedB(query)#u: torch.Size([32, 3, 3])
        #print("u: ", u.shape)

        #reshape u from 32x3x3 to 32x9 to perform matrix multiplication with a_embed
        u = u.reshape(32,9) #u: torch.Size([32,9])

        #reshape a_embed from 32x6x3 to 32x18 to perform matrix multiplication with u
        a_embed = a_embed.reshape(32,18) #a_embed: torch.Size([32,18])

        inner_product = torch.matmul(torch.t(u), a_embed) #inner_product: torch.Size([9, 18])
        #print("inner_product:", inner_product.shape)

        p = torch.softmax(inner_product.float(), dim=-1) #p: torch.Size([9, 18])
        #print("p: ", p.shape)

        ci = torch.matmul(p, torch.t(a_embed)) #ci: torch.Size([9, 32])
        #print("ci: ", ci.shape)

        o = torch.sum(ci, 0) #o: torch.Size([32])
        #print("o: ", o.shape)

        ten3 = torch.add(o, torch.t(u)) #ten3: torch.Size([9,32])
        #print("ten3: ", ten3.shape)

        #reshape ten3 from 9x32 to 96x3 so it matches the linear layer, LinW of size 3x20
        ten3 = ten3.reshape(96,3)

        ten4 = self.LinW(ten3) #ten4: torch.Size([96, 20])
        #print("ten4: ", ten4.shape)

        a = torch.softmax(ten4, dim=0) #a: torch.Size([96, 20])
        #print("a: ", a.shape)

        #torch.max to get one tensor
        #torch.mul it with vocab (20)
        #torch.round to round off the value
        return torch.round(torch.mul(torch.max(a), V))

#Initialize the model
model = EMN()

#Optimizer and Loss function
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i, batch in enumerate(train_iter):
    if i < len(train_iter)-1:

        story = batch.story
        query = batch.query
        answer = batch.answer

        optimizer.zero_grad()
        output = model(story, query)
        l = loss(output.type(torch.float32), answer[i].type(torch.float32))
        l.backward()
        optimizer.step()

        # print("answer: ", answer[0].item()) #torch.Size([32, 1])
        # print("output: ", output.item())#torch.Size([])

        if i <= len(train_iter):
            print(f'Batch no. [{i+1}], Loss: {l.item():.4f}')
    else:
        break









