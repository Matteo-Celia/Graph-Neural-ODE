import torch
import torch.nn as nn
import blocks
import torch.nn.functional as F
import numpy as np
from random import randint
from graphs import GraphsTuple


def data_dicts_to_graphs_tuple(graph_dicts:dict):
    for k,v in graph_dicts.items():
        graph_dicts[k]=torch.tensor(v)
    return GraphsTuple(**graph_dicts)

def pbc_diff(pos1, pos2, box_size=6):
    diff = pos1 - pos2
    # Periodic boundry conditions
    diff[diff > box_size/2] = diff[diff > box_size/2] - box_size 
    diff[diff <= -box_size/2] = diff[diff <= -box_size/2] + box_size
    return diff