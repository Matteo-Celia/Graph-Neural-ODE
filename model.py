
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
        

    def forward(self, t, h, **kwargs): #, args
        #rebuild matrix
        #h = h.squeeze(0) # shape (T,N*D)
        
        nodes = h.reshape(-1,self.n_particles,self.nodedim) # shape (T,N,D)
        #print(nodes.shape)
        #recompute graph based on h
        distances, R_s, R_r = build_senders_receivers(nodes)
        self.graph = build_GraphTuple(nodes, distances, R_s, R_r)
        device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
        #print(device)
        for node in self.graph.nodes:
            node.to(device)
        self.graph.nodes.to(device)
        self.graph.edges.to(device)
        self.graph.receivers.to(device)
        self.graph.senders.to(device)
        new_nodes = self._node_block(self._edge_block(self.graph)).nodes # shape (N_nodes*traj_len,N_features)
        #batch nodes' features as (trajectory_len,num_nodes*nodedim)
        new_h = new_nodes.reshape(-1, self.n_particles*self.nodedim) # shape (T,N*D)

        return new_h



class UpdateFunction(nn.Module):
    def __init__(self, featdim):
        super(UpdateFunction,self).__init__()
        self.t = 0
        self.Dt = None
        self.nn = nn.Sequential(
        nn.Linear(featdim, 64),
        nn.Tanh(), 
        nn.Linear(64, featdim))

    def forward(self, t, x, **kwargs ): #**kwargs

        return self.Dt + (t-self.t)*self.nn(x)
         

class GNSTODE(nn.Module):
    #at each forward pass:
    
    
    def __init__(self, n_particles, space_int=50, temp_int=50, box_size=6, integrator='euler', simulation_type='gravity'):
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
        
        self.featdim = n_particles*nodedim
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
        self.spatial_model = NeuralODE(self.gin, sensitivity='adjoint', solver=self.integrator, interpolator=None, return_t_eval=False).to(self.device) #atol=1e-3, rtol=1e-3, 
        self.temporal_model = NeuralODE(self.F, sensitivity='adjoint', solver=self.integrator, interpolator=None, return_t_eval=False).to(self.device) #atol=1e-3, rtol=1e-3,
        
        

    def forward(self, input_trajectory, dt):
        
        Xt = input_trajectory #shape (T,N,D)
        Xt = Xt.squeeze(0)
        #print("GNSTODE forward")
        #spatial processing
        #reshape Xt as (traj_len,N_nodes*N_features)
        Xt = Xt.reshape(-1,Xt.shape[-2]*Xt.shape[-1])
        HL = self.spatial_model(Xt,self.L_span)
        
        ##split matrix based on the nodes of each graph and then flatten to build a matrix: (trajectory_len,num_nodes*nodedim) 
        #HL_split = split_matrix_np(HL,len(num_nodes), self.n_particles) 
        #print(HL.shape)
        Dt = self.NN(HL[-1]) #get just the final solution

        #print(Dt.shape)
        #temporal processing
        self.F.Dt = Dt
        
        Xtpreds = self.temporal_model(Xt,self.t_span)
        #get just the final solution and reshape it as (T,N,D) again
        Xtpreds = Xtpreds[-1].reshape(-1,input_trajectory.shape[-2],input_trajectory.shape[-1])
        #print(Xtpreds.shape)

        # for i,t in enumerate(self.traj_len):
            
        #     self.temporal_model = NeuralODE(F, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(self.device)
        #     Xtpred = self.temporal_model(Xt[i],self.t_span)
        #     Xtpreds.append(Xtpred)

        return Xtpreds
