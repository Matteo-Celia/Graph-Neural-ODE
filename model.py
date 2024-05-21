
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
from utils import split_matrix_np, build_GraphTuple, build_senders_receivers

class GraphInteractionNetwork(nn.Module):
    def __init__(self, n_particles, nodedim, edgedim):
        super(GraphInteractionNetwork,self).__init__()
        self.graph = None
        self.n_particles = n_particles
        self.nodedim = nodedim
        self._edgedim = edgedim
        self._edge_block = blocks.EdgeBlock(nodedim, edgedim, use_globals=False)
        self._node_block = blocks.NodeBlock(nodedim, edgedim, use_sent_edges=False, use_globals=False)
        

    def forward(self, t, h):

        #recompute graph based on h
        #nodes = h.reshape(-1,self.nodedim)
        R_s, R_r = build_senders_receivers(h)
        self.graph = build_GraphTuple(h, R_s, R_r)
        
        new_nodes = self._node_block(self._edge_block(self.graph)).nodes
        return split_matrix_np(new_nodes,len(self.graph.n_node), self.n_particles)



class UpdateFunction(nn.Module):
    def __init__(self, Dt, featdim):
        super(UpdateFunction,self).__init__()
        self.t = 0
        self.Dt = Dt
        self.linear = nn.Linear(featdim, featdim)

    def forward(self, tao, h):

        return self.Dt + (self.t-tao)*self.linear(h)
         

class GNSTODE(nn.module):
    #at each forward pass:
    
    
    def __init__(self, n_particles, space_int=100, temp_int=100, box_size=6, integrator='rk4', simulation_type='gravity'):
        super(GNSTODE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            nodedim = 6 # (mass, x, y, charge, px, py)
        else:
            nodedim = 5 # (mass, x, y, px, py)
        edgedim = 1 #distance between particles
        self.n_particles = n_particles
        # Set integrator to use
        self.integrator = integrator

        self.L_span = torch.linspace(0, 1, space_int)
        self.t_span = torch.linspace(0, 1, temp_int)
        
        self.featdim = nodedim*n_particles
        # Linear layer to obtain Dt from H_L
        #self.linear = nn.Linear(self.featdim, self.featdim)
        
        # Set box size
        self.box_size = box_size

        self.gin = GraphInteractionNetwork(n_particles,nodedim, edgedim)
        self.NN = nn.Sequential(
        nn.Linear(self.featdim, 64),
        nn.Tanh(), 
        nn.Linear(64, self.featdim))
 
        self.F = UpdateFunction(featdim=self.featdim)
        

    def forward(self, input_trajectory, dt):#change just inputs,R_s and R_r graph is built inside the gin
        
        #num_nodes = input_trajectory.shape[1]
        
        Xt = input_trajectory
        #spatial processing
        self.spatial_model = NeuralODE(self.gin, sensitivity='adjoint', solver=self.integrator, interpolator=None, atol=1e-3, rtol=1e-3).to(self.device)
        #maybe needed to batch Xr before NODE the same way HL is batched afterwards
        HL = self.spatial_model(Xt,self.L_span)
        
        ##split matrix based on the nodes of each graph and then flatten to build a matrix: (trajectory_len,num_nodes*nodedim) 
        #HL_split = split_matrix_np(HL,len(num_nodes), self.n_particles) 
        
        Dt = self.NN(HL)

        Xtpreds = []

        #temporal processing
        self.F.Dt = Dt
        #Xt_split = split_matrix_np(Xt,len(num_nodes), self.n_particles)
        self.temporal_model = NeuralODE(self.F, sensitivity='adjoint', solver=self.integrator, interpolator=None, atol=1e-3, rtol=1e-3).to(self.device)
        Xtpred = self.temporal_model(Xt,self.t_span)
        Xtpreds.append(Xtpred)

        # for i,t in enumerate(self.traj_len):
            
        #     self.temporal_model = NeuralODE(F, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(self.device)
        #     Xtpred = self.temporal_model(Xt[i],self.t_span)
        #     Xtpreds.append(Xtpred)

        return np.array(Xtpreds)
