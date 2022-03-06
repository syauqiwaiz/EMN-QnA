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
train_iter, valid_iter, test_iter, vocab = dataloader(batch_size=32, memory_size=50, task=15, joint=False, tenK=False)


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

        ext_memory = story
        ext_memory = self.embedA(ext_memory)

        a_embed = torch.sum(ext_memory, dim=2)

        u = self.embedB(query)

        u = torch.sum(u, dim=1)

        inner_product = torch.einsum('ijk, ik->j', a_embed, u)

        p = torch.softmax(inner_product.float(), dim=-1)

        ci = torch.einsum('ijk, j->ijk',a_embed, p)

        o = torch.sum(ci, dim=1)

        ten3 = torch.add(o, u)

        ten4 = self.LinW(ten3)

        a = torch.softmax(ten4, dim=1)

        return a

#Initialize the model
model = EMN()

#Optimizer and Loss function
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Training the model

total_loss = 0
for i, batch in enumerate(train_iter):

    story = batch.story
    query = batch.query
    answer = batch.answer

    optimizer.zero_grad()
    output = model(story, query)
    l = loss(output.float(), answer.squeeze(1))
    l.backward()
    optimizer.step()

    total_loss += l

    print(f'Batch no. [{i+1}], Loss: {l.item():.3f}')

print(f'Average Training Loss: {total_loss/len(train_iter):.2f}')









