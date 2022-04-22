#Implemented in PyCharm IDE using Python 3.9.1
#Machine Learning framework used is PyTorch 1.8.0
#PyTorch "torchtext" version 0.9.0

#import necessary libraries
import os
import torch
import torch.nn as nn
from torchtext.legacy.datasets import BABI20
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

#argument parser
parser = argparse.ArgumentParser(description="Training the model")
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--num_hops", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--memory_size", type=int, default=50)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--max_clip", type=float, default=40.0)
parser.add_argument("--embed_dim", type=int, default=20)
parser.add_argument("--tensorboard", type=bool, default=False)
parser.add_argument("--tenK", type=bool, default=False)
parser.add_argument("--save", type=bool, default=False)
args = parser.parse_args()

#create a function to wrap the iteration of bAbI dataset and load it
def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK,
        device=torch.device("cpu"), shuffle=True)
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab

#design model
class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create embedding layers and linear layer
        self.embedA = nn.Embedding(V, d, padding_idx=0)
        self.embedB = nn.Embedding(V, d, padding_idx=0)
        self.embedC = nn.Embedding(V, d, padding_idx=0)
        self.LinW = nn.Linear(d, V)
        self.dropout = nn.Dropout()

        #Initialize weights
        self.embedA.weight.data.normal_(0, 0.1)
        self.embedB.weight.data.normal_(0, 0.1)
        self.embedC.weight.data.normal_(0, 0.1)
        self.LinW.weight.data.normal_(-0.1, 0.1)


    def forward(self, story, query):

        self.story = story
        self.query = query

        u = self.dropout(self.embedB(self.query))
        u = torch.sum(u, dim=1)

        for k in range(num_hops):

            a_embed = self.dropout(self.embedA(self.story))
            a_embed = torch.sum(a_embed, dim=2)

            c_embed = self.dropout(self.embedC(self.story))
            c = torch.sum(c_embed, dim=2)

            ip = torch.bmm(a_embed, u.unsqueeze(2)).squeeze()

            p = torch.softmax(ip, -1).unsqueeze(1)

            o = torch.bmm(p, c).squeeze(1)

            u = o + u

        a = self.LinW(u)

        return a

#Create function for loss visualization
def tensor_board1(loss, epoch):
    return train_writer.add_scalar("Loss/epoch", loss, epoch)
def tensor_board2(loss, epoch):
    return valid_writer.add_scalar("Loss/epoch", loss, epoch)


if args.tensorboard == True:
    train_writer = SummaryWriter(os.path.join("./runs", "train"))
    valid_writer = SummaryWriter(os.path.join("./runs", "valid"))

#Train and validate model
def train(EPOCH, loss, lr, optim):

    optimizer = optim(model.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    min_valid_loss = np.inf

    for epoch in range(EPOCH):

        total_loss = 0

        for i, batch in enumerate(train_iter):

            story = batch.story
            query = batch.query
            answer = batch.answer

            optimizer.zero_grad()
            output = model(story, query)
            l = loss(output, answer.squeeze(1))
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_clip)
            optimizer.step()

            total_loss += l.item()
            avg_tloss = total_loss / len(train_iter)



        valid_loss = 0.0
        model.eval()
        for i, batch in enumerate(valid_iter):
            story = batch.story
            query = batch.query
            answer = batch.answer

            output = model(story, query)
            l = loss(output, answer.squeeze(1))
            valid_loss += l.item()
            avg_vloss = valid_loss / len(valid_iter)

        if args.tensorboard == True:
            tensor_board1(avg_tloss, epoch)
            tensor_board2(avg_vloss, epoch)

        if epoch % 10== 0:
            print(f'Epoch [{epoch}] \t\t Training Loss: {total_loss / len((train_iter)):.3f} \t\t Validation Loss: {valid_loss / len(valid_iter):.3f}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.3f}--->{valid_loss:.3f})')
                min_valid_loss = valid_loss
        scheduler.step()

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

        print(f"Accuracy of Task {args.task} with {args.epochs} epochs and {args.lr} learning rate: {((n_correct/n_answers)*100):.2f}%")
        #print(f"Error rate: {((1 - (n_correct/n_answers))*100):.2f}%")



# call functions
if __name__ == '__main__':

    #using defined dataloader function, split the data to train, valid, test and obtain the vocab
    train_iter, valid_iter, test_iter, vocab = dataloader(batch_size=args.batch_size, memory_size=args.memory_size,
                                                          task=args.task, joint=False, tenK=args.tenK)


    #Variables for layers defined
    V = len(vocab)
    d = args.embed_dim
    num_hops = args.num_hops

    #Initialize model
    model = EMN()


    train(EPOCH=args.epochs, loss= nn.CrossEntropyLoss(), lr= args.lr, optim= torch.optim.SGD)
    test()

if args.tensorboard == True:
    train_writer.flush()
    valid_writer.flush()
    train_writer.close()
    valid_writer.close()

#Save model
if args.save == True:
    torch.save(model.state_dict(), "EMN.pth")


