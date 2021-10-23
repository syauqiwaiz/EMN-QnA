#EMN implementation to solve BABI tasks based on https://github.com/zshihang/MemN2N

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from model import MemN2N
from helpers import dataloader, get_fname, get_params

def train(train_iter, model, optimizer, epochs, max_clip, valid_iter=None):
   total_loss = 0
   valid_data = list(valid_iter)
   valid_loss = None
   next_epoch_to_report = 5
   pad = model.vocab.stoi['<pad>']
   
