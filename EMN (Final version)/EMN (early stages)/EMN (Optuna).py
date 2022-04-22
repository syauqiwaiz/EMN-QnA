import torch
import torch.nn as nn
from torchtext.legacy.datasets import BABI20
import torch.optim as optim
import optuna
import argparse

parser = argparse.ArgumentParser(description="Training the model")
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--num_hops", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--memory_size", type=int, default=50)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--max_clip", type=float, default=40.0)
parser.add_argument("--embed_dim", type=int, default=20)
parser.add_argument("--tenK", type=bool, default=False)
parser.add_argument("--trials", type=int, default=30)
args = parser.parse_args()

#dataset
def dataloader(batch_size, memory_size, task, joint, tenK):
    train_iter, valid_iter, test_iter = BABI20.iters(
        batch_size=batch_size, memory_size=memory_size, task=task, joint=joint, tenK=tenK, device=torch.device("cpu"))
    return train_iter, valid_iter, test_iter, train_iter.dataset.fields['query'].vocab

train_iter, valid_iter, test_iter, vocab = dataloader(batch_size=args.batch_size, memory_size=args.memory_size,
                                                          task=args.task, joint=False, tenK=args.tenK)

V = len(vocab)
d = args.embed_dim
num_hops = args.num_hops


class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create Embeddings
        self.embedA = nn.Embedding(V, d, padding_idx=0)
        self.embedB = nn.Embedding(V, d, padding_idx=0)
        self.embedC = nn.Embedding(V, d, padding_idx=0)
        self.LinW = nn.Linear(d, V)


        self.embedA.weight.data.normal_(0, 0.1)
        self.embedB.weight.data.normal_(0, 0.1)
        self.embedC.weight.data.normal_(0, 0.1)
        self.LinW.weight.data.normal_(0.25, 0.25)


    def forward(self, story, query):

        self.story = story
        self.query = query

        u = self.embedB(self.query)
        u = torch.sum(u, dim=1)

        for k in range(num_hops):

            a_embed = self.embedA(self.story)
            a_embed = torch.sum(a_embed, dim=2)

            c_embed = self.embedC(self.story)
            c = torch.sum(c_embed, dim=2)

            ip = torch.bmm(a_embed, u.unsqueeze(2)).squeeze()

            p = torch.softmax(ip, -1).unsqueeze(1)

            o = torch.bmm(p, c).squeeze(1)
            u = o + u

        a = self.LinW(u)

        return a

#Initialize the model
model = EMN()

EPOCH = args.epochs

def train_and_evaluate(param, model):

    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])
    loss = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):

        for i, batch in enumerate(train_iter):
            story = batch.story
            query = batch.query
            answer = batch.answer

            optimizer.zero_grad()
            output = model(story, query)
            l = loss(output.float(), answer.squeeze(1))
            l.backward()
            optimizer.step()

    model.eval()
    total_error = 0

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

            loss = model(story, query)
            _, loss = torch.max(loss, -1)
            total_error += torch.mean((loss != answer.view(-1)).float()).item()

        accuracy = (n_correct / n_answers) * 100

    return accuracy

def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'optimizer': trial.suggest_categorical("optimizer", ["SGD", "ASGD"]),
    }

    accuracy = train_and_evaluate(params, model)

    return accuracy

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=args.trials)

    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


















