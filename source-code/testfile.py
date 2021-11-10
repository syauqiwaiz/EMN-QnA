#EMN implementation to solve BABI tasks based on https://github.com/zshihang/MemN2N

import os

import torch  # use PyTorch
import torch.nn as nn  # contain classes for neural network
import torch.nn.functional as F  # contain functions for neural network
from torch import optim  # to implement optimization algorithm during training

from model import MemN2N  # import created model MemN2N from model.py
from helpers import dataloader, get_fname, get_params  # import functions created and defined from helpers.py

# train the model with the arguments:train_iter, model, optimizer, epochs, max_clip and valid_iter
# train_iter is training iteration variable
# model is the model imported MemN2N
# optimizer is calling the optimization algorithm
# epoch is defined as the completion of all training datasets passed thru machine learning algorithm
# max_clip is a variable defined for the clipping of the gradient value if it exceeds an expected range
# valid_iter is set to default which is none
# the output is measured using validation loss
# total_loss = 0 and valid_loss = None are for initialization
# valid_iter converted to a list and stored in the variable, valid_data
# next_epoch_to_report=5 is for the model to report train progress every 5 epochs completed
# pad = model.vocab.stoi['<pad>'] is mapping token strings (the model) to numerical identifiers.


def train(train_iter, model, optimizer, epochs, max_clip, valid_iter=None):
    total_loss = 0
    valid_data = list(valid_iter)
    valid_loss = None
    next_epoch_to_report = 5
    pad = model.vocab.stoi['<pad>']

# Using for loop and initialized the story, query and answer in batches
# optimizer.zero_grad() is essentially resetting the gradients to zero before doing backpropagation
# output is the story and query
# nll_loss function is the negative log likelihood loss, in this case it is the sum of outputs and answer.view(-1)
# loss.backward() is called to perform loss function on the loss variable
# nn.utils.clip_grad_norm_() is clipping the gradient of the model to specific value
# optimizer.step iterate over all parameters (tensors) and update their values
# total_loss value update with the loss that was calculated.
# if the model uses linear then the loss is reset to 0 and the training method began to get new loss value
# the model then report train progress every 5 epochs which is the average batch loss
    for _, batch in enumerate(train_iter, start=1):
        story = batch.story
        query = batch.query
        answer = batch.answer
        

        optimizer.zero_grad()
        outputs = model(story.cuda(), query.cuda())
        loss = F.nll_loss(outputs, answer.view(-1).cuda(), ignore_index=pad, reduction='sum')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_clip)
        optimizer.step()
        total_loss += loss.item()

        # linear start
        if model.use_ls:
            loss = 0
            for k, batch in enumerate(valid_data, start=1):
                story = batch.story
                query = batch.query
                answer = batch.answer
                outputs = model(story.cuda(), query.cuda())
                loss += F.nll_loss(outputs, answer.view(-1).cuda(), ignore_index=pad, reduction='sum').item()
            loss = loss / k
            if valid_loss and valid_loss <= loss:
                model.use_ls = False
            else:
                valid_loss = loss


        if train_iter.iterations == next_epoch_to_report:
            print("#! epoch {:d} average batch loss: {:5.4f}".format(
                int(train_iter.iterations), total_loss / len(train_iter)))
            next_epoch_to_report += 5
        if int(train_iter.iterations) == train_iter.epoch:
            total_loss = 0
        if train_iter.iterations == epochs:
            break



# This is the testing phase where the error is calculated


def eval(test_iter, model):
    total_error = 0
    for k, batch in enumerate(test_iter, start=1):
        story = batch.story
        query = batch.query
        answer = batch.answer
        outputs = model(story.cuda(), query.cuda())
        _, outputs = torch.max(outputs, -1)
        total_error += torch.mean((outputs.cuda() != answer.view(-1).cuda()).float()).item()
    print("#! average error: {:5.1f}".format(total_error / k * 100))

#essentially a method to call above methods(train, test)

def run(config):
    print("#! preparing data...")
    train_iter, valid_iter, test_iter, vocab = dataloader(config.batch_size, config.memory_size,
                                                          config.task, config.joint, config.tenk)
    #print(train_iter.)
    print("#! instantiating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemN2N(get_params(config), vocab).to(device)

    if config.file:
        with open(os.path.join(config.save_dir, config.file), 'rb') as f:
            if torch.cuda.is_available():
                state_dict = torch.load(f, map_location=lambda storage, loc: storage.cuda())
            else:
                state_dict = torch.load(f, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)


    if config.train:
        print("#! training...")
        optimizer = optim.Adam(model.parameters(), config.lr)
        train(train_iter, model, optimizer, config.num_epochs, config.max_clip, valid_iter)
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        torch.save(model.state_dict(), os.path.join(config.save_dir, get_fname(config)))

    print("#! testing...")
    with torch.no_grad():
        eval(test_iter, model)
        


       
    





###################
###################
#################
# RUNNING MAIN HERE

from collections import namedtuple


config = namedtuple('config',['train','save_dir','file','num_epochs',
                                'batch_size','lr','embed_size','task','memory_size',
                                'num_hops','max_clip','joint','tenk','use_bow','use_lw','use_ls'])
config.train = True
config.save_dir = '.save'
config.file = ''
config.num_epochs = 1000
config.batch_size = 32
config.lr = 0.02
config.embed_size = 20
config.task = 1
config.memory_size = 50
config.num_hops = 3
config.max_clip = 40.0
config.joint = False
config.tenk = False
config.use_bow = False
config.use_lw = False
config.use_ls = False

run(config)
#@click.command()
#@click.option('--train', default=True, is_flag=True, help="Train phase.")
#@click.option('--save_dir', default='.save', help="Directory of saved object files.",
#              show_default=True)
#@click.option('--file', default='', help="Path of saved object file to load.")
#@click.option('--num_epochs', type=int, default=10000, help="Number of epochs to train.",
#              show_default=True)
#@click.option('--batch_size', type=int, default=32, help="Batch size.", show_default=True)
#@click.option('--lr', type=float, default=0.02, help="Learning rate.", show_default=True)
#@click.option('--embed_size', type=int, default=20, help="Embedding size.",
#              show_default=True)
#@click.option('--task', type=int, default=1, help="Number of task to learn.",
#              show_default=True)
#@click.option('--memory_size', type=int, default=50, help="Capacity of memory.",
#              show_default=True)
#@click.option('--num_hops', type=int, default=3, help="Embedding size.", show_default=True)
#@click.option('--max_clip', type=float, default=40.0, help="Max gradient norm to clip",
#              show_default=True)
#@click.option('--joint', is_flag=True, help="Joint learning.")
#@click.option('--tenk', is_flag=True, help="Use 10K dataset.")
#@click.option('--use_bow', is_flag=True, help="Use BoW, or PE sentence representation.")
#@click.option('--use_lw', is_flag=True, help="Use layer-wise, or adjacent weight tying.")
#@click.option('--use_ls', is_flag=True, help="Use linear start.")
# def cli(**kwargs):
#     config = namedtuple("config", kwargs.keys())(**kwargs)
#     run(config)


# if __name__ == "__main__":
#     cli()

