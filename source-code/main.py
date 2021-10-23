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
          
     if train_iter.epoch == next_epoch_to_report:
        print("#! epoch {:d} average batch loss: {:5.4f}".format(
            int(train_iter.epoch), total_loss / len(train_iter)))
        next_epoch_to_report += 5
     if int(train_iter.epoch) == train_iter.epoch:
        total_loss = 0
     if train_iter.epoch == epochs:
            break
         
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
       
        
