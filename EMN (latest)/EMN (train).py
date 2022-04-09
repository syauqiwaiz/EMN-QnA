import torch
import torch.nn as nn
from torchtext.legacy.datasets import BABI20
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter()

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK, device=torch.device("cpu"))
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab

# Use dataloader function created to split train, test, valid and obtain vocab
train_iter, valid_iter, test_iter, vocab = dataloader(batch_size=32, memory_size=50, task=20, joint=False, tenK=False)

# Parameters
V = len(vocab)
d = 20


class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create Embeddings
        self.embedA = nn.Embedding(V, d)
        self.embedB = nn.Embedding(V, d)
        self.embedC = nn.Embedding(V, d)
        self.LinW = nn.Linear(d, V)

    def forward(self, story, query):

        self.story = story
        self.query = query

        a_embed = self.embedA(self.story)
        a_embed = torch.sum(a_embed, dim=2)

        u = self.embedB(self.query)
        u = torch.sum(u, dim=1)

        inner_product = torch.einsum('ijk, ik->j', a_embed, u)

        p = torch.softmax(inner_product.float(), dim=-1)

        c_embed = self.embedC(self.story)
        c_embed = torch.sum(c_embed, dim=2)

        ci = torch.einsum('ijk, j->ijk', c_embed, p)

        o = torch.sum(ci, dim=1)

        a = self.LinW(torch.add(o, u))

        return a

#Initialize the model
model = EMN()

#Train the model

#def tensor_board(loss, epoch):
    #return writer.add_scalar("Loss/epoch", loss, epoch)



def train(EPOCH, loss, lr, optim):

    optimizer = optim(model.parameters(), lr)
    min_valid_loss = np.inf

    for epoch in range(EPOCH):

        total_loss = 0

        for i, batch in enumerate(train_iter):

            story = batch.story
            query = batch.query
            answer = batch.answer

            optimizer.zero_grad()
            output = model(story, query)
            l = loss(output, answer.squeeze(1)).to(device)
            l.backward()
            optimizer.step()

            total_loss += l.item()
            avg_loss = total_loss / len(train_iter)

        valid_loss = 0.0
        model.eval()
        for i, batch in enumerate(valid_iter):
            story = batch.story
            query = batch.query
            answer = batch.answer

            output = model(story, query)
            l = loss(output, answer.squeeze(1)).to(device)
            valid_loss += l.item()

        #tensor_board(avg_loss, epoch)

        if epoch % 10== 0:
            print(f'Epoch [{epoch + 1}] \t\t Training Loss: {total_loss / len((train_iter)):.3f} \t\t Validation Loss: {valid_loss / len(valid_iter):.3f}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.3f}--->{valid_loss:.3f})')
                min_valid_loss = valid_loss


#Testing the model

def test():

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_answers = 0

        for i, batch in enumerate(test_iter):

            story = batch.story
            query = batch.query
            answer = batch.answer

            output = model(story, query)
            for index, i in enumerate(output):
                if torch.argmax(i) == answer[index]:
                    n_correct += 1
                n_answers += 1


        print(f"Accuracy: {((n_correct/n_answers)*100):.2f}%")
        print(f"Error rate: {((1 - (n_correct/n_answers))*100):.2f}%")

# call functions

train(EPOCH=100, loss= nn.CrossEntropyLoss(), lr= 0.1, optim= torch.optim.SGD)
test()

#writer.flush()
#writer.close()

#Save model

#torch.save(model.state_dict(), "EMN20.pth")


















