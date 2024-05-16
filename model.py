
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import blocks
from random import randint
from graphs import GraphsTuple

class GraphInteractionNetwork(nn.Module):
    def __init__(self,graph):
        super(GraphInteractionNetwork,self).__init__()
        self.graph = graph
        self._edge_block = blocks.EdgeBlock(graph, use_globals=False)
        self._node_block = blocks.NodeBlock(graph, use_sent_edges=False, use_globals=False)
        

    def forward(self, t, h):

        self.graph.replace(nodes=h)
        return self._node_block(self._edge_block(self.graph)).nodes



class UpdateFunction(nn.Module):
    def __init__(self, t, Dt, featdim):
        super(UpdateFunction,self).__init__()
        self.t = t
        self.Dt = Dt
        self.linear = nn.Linear(featdim, featdim)

    def forward(self, tao, h):

        return self.Dt + (self.t-tao)*self.linear(h)
         

class GNSTODE(nn.module):
    #at each forward pass:
    
    
    def __init__(self, n_particles, traj_len, space_int=100, temp_int=100, box_size=6, integrator='rk4', simulation_type='gravity'):
        super(GNSTODE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            node_input_dim = 4 # (mass, charge, px, py)
        else:
            node_input_dim = 3 # (mass, px, py)
        self.L_span = torch.linspace(0, 1, space_int)
        self.t_span = torch.linspace(0, 1, temp_int)
        self.gin = None
        T = len(traj_len)
        self.featdim = node_input_dim*n_particles*T
        # Linear layer to obtain Dt from H_L
        self.linear = nn.Linear(self.featdim, self.featdim)
        self.featdimTraj = node_input_dim*n_particles
        # Set box size
        self.box_size = box_size
        self.NN = nn.Sequential(
        nn.Linear(self.featdimTraj, 64),
        nn.Tanh(), 
        nn.Linear(64, self.featdimTraj))
        # Set integrator to use
        self.integrator = integrator

    def forward(self, graph):
        
        Xt = graph.nodes
        self.gin = GraphInteractionNetwork(graph)
        
        self.spatial_model = NeuralODE(self.gin, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(self.device)
        H = self.spatial_model(Xt,self.L_span)
        Dt = []

        for i in H.shape[0]:
            Dt.append(self.NN(H[i]))

        Xtpreds = []

        for i,t in enumerate(self.traj_len):
            F = UpdateFunction(t,Dt[i],self.featdim)
            self.temporal_model = NeuralODE(F, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(self.device)
            Xtpred = self.temporal_model(Xt[i],self.t_span)
            Xtpreds.append(Xtpred)

        return Xtpreds
