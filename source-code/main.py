#EMN implementation to solve BABI tasks based on https://github.com/zshihang/MemN2N

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from model import MemN2N
from helpers import dataloader, get_fname, get_params

 
