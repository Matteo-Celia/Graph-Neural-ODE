
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

        self._edge_block = blocks.EdgeBlock(graph, use_globals=False)
        self._node_block = blocks.NodeBlock(graph, use_sent_edges=False, use_globals=False)
        

    def forward(self, graph):

        return self._node_block(self._edge_block(graph))


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()
        
        self.output_size = output_size
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        batch_size, n_relations, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.layers(x)
        x = x.view(batch_size, n_relations, self.output_size)
        return x
    

class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ObjectModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2), #speedX and speedY
        )
        
    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_objects, input_size]
        Returns:
            [batch_size * n_objects, 2] speedX and speedY
        '''
        input_size = x.size(2)
        x = x.view(-1, input_size)
        return self.layers(x)

class InteractionNetwork(nn.Module):
    def __init__(self, object_dim, n_relations, relation_dim, effect_dim): #num_objects
        super(InteractionNetwork, self).__init__()
        
        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, 150)
        self.object_model     = ObjectModel(object_dim + effect_dim, 100)
    
    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        senders   = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], 2))
        effect_receivers = receiver_relations.bmm(effects)
        predicted = self.object_model(torch.cat([objects, effect_receivers], 2))
        return predicted


class ODEFunc(nn.Module):
    def __init__(self, net:nn.Module):
        
        super().__init__()
        self.net = net
        self.nfe = 0
            
    def forward(self, t, x):
        self.nfe += 1
        x = self.net(x)
        return x
    

class ODEBlock(nn.Module):
    def __init__(self, odefunc:nn.Module, method:str='rk4', rtol:float=1e-3, atol:float=1e-4, adjoint:bool=True):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol

    def forward(self, x:torch.Tensor, T:int=1):
        self.integration_time = torch.tensor([0, T]).float()
        self.integration_time = self.integration_time.type_as(x)

        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
            
        return out[-1]

class HOGN(BaseIntegratorModel):
    
    def __init__(self, box_size=6, edge_output_dim=150, node_output_dim=100, global_output_dim=100, integrator='rk4', simulation_type='gravity'):
        super(HOGN, self).__init__()

        if edge_output_dim < 1:
            edge_output_dim = 150
        if node_output_dim < 1:
            node_output_dim = 100 
        if global_output_dim < 1:
            global_output_dim = 100 

        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            node_input_dim = 4 # (mass, charge, px, py)
        else:
            node_input_dim = 3 # (mass, px, py)

        self.edge_model = EdgeModel(input_dim=2*node_input_dim+2, output_dim=edge_output_dim, softplus=True, box_size=box_size) # input dim: sender and reciever node features  + disntace vector

        self.node_model = NodeModel(input_dim=node_input_dim+edge_output_dim, output_dim=node_output_dim, softplus=True) # input dim: input node features + embedded edge features

        self.global_model = GlobalModel(input_dim=node_output_dim+edge_output_dim, output_dim=global_output_dim) # input dim: embedded node features and embedded edge features

        # Linear layer to transform global embeddings to a Hamiltonian
        self.linear = nn.Linear(global_output_dim, 1)

        # Set box size
        self.box_size = box_size

        # Set integrator to use
        self.integrator = integrator

    # Here vertices V are in canonical coordinates [x,y,px,py]
    def forward_step(self, mass_charge, V, R_s, R_r):

        # Drop position from particles/nodes and add mass and charge (if present)
        V_no_pos = torch.cat([mass_charge, V[:,:,2:]], dim=2)

        R_s = R_s.unsqueeze(2)
        R_r = R_r.unsqueeze(2)

        # Edge block
        E_n = self.edge_model(V_no_pos, V[:,:,:2], R_s, R_r)

        # Node block
        V_n = self.node_model(V_no_pos, E_n)

        # Global block
        U_n = self.global_model(V_n, E_n)

        # Hamiltonian
        H = self.linear(U_n)

        # Hamiltonian derivatives w.r.t inputs = dH/dq dH/dp
        partial_derivatives = torch.autograd.grad(H.sum(), V, create_graph=True)[0] #, only_inputs=True

        # Return dq and dp
        return torch.cat([partial_derivatives[:,:,2:], partial_derivatives[:,:,:2] * (-1.0)], dim=2)  # dq=dH/dp, dp=-dH/dq

    def forward(self, state, R_s, R_r, dt):
        # Transform inputs [m, x, y, vx, vy] to canonical coordinates [x,y,px,py]
        mass_charge = state[:,:,:-4] # if no charge = [m]; with charge = [m, c]
        momentum = state[:,:,-2:] * mass_charge[:,:,0].unsqueeze(2)
        V = torch.cat([state[:,:,-4:-2], momentum], dim=2)
        # Require grad to be able to compute partial derivatives
        if not V.requires_grad:
            V.requires_grad = True
        
        # Compute updated canonical coordinates
        if self.integrator == 'rk4':
            new_canonical_coordinates = self.rk4(dt, mass_charge, V, R_s, R_r)
        elif self.integrator == 'euler':
            new_canonical_coordinates = self.euler(dt, mass_charge, V, R_s, R_r)
        else:
            raise Exception
        
        # Convert back to original state format [x, y, vx, vy]
        velocity = torch.div(new_canonical_coordinates[:,:,2:], mass_charge[:,:,0].unsqueeze(2))
        new_state = torch.cat([new_canonical_coordinates[:,:,:2], velocity], dim=2)
        return new_state
    
class GNSTODE(nn.module):
    #at each forward pass:
    
    
    def __init__(self, box_size=6, edge_output_dim=150, node_output_dim=100, global_output_dim=100, integrator='rk4', simulation_type='gravity'):
        super(GNSTODE, self).__init__()

        if edge_output_dim < 1:
            edge_output_dim = 150
        if node_output_dim < 1:
            node_output_dim = 100 
        if global_output_dim < 1:
            global_output_dim = 100 

        self.simulation_type = simulation_type
        # Set number of node features, excluding the position (x,y)
        if self.simulation_type == 'coulomb':
            node_input_dim = 4 # (mass, charge, px, py)
        else:
            node_input_dim = 3 # (mass, px, py)

        self.gin = InteractionNetwork()
        
        self.spatial_model = ODEBlock(input_dim=2*node_input_dim+2, output_dim=edge_output_dim, softplus=True, box_size=box_size) # input dim: sender and reciever node features  + disntace vector

        self.temporal_model = NodeModel(input_dim=node_input_dim+edge_output_dim, output_dim=node_output_dim, softplus=True) # input dim: input node features + embedded edge features


        # Linear layer to transform global embeddings to a Hamiltonian
        self.linear = nn.Linear(global_output_dim, 1)

        # Set box size
        self.box_size = box_size

        # Set integrator to use
        self.integrator = integrator

    def forward(self, state, R_s, R_r, dt):
        # Transform inputs [m, x, y, vx, vy] to canonical coordinates [x,y,px,py]
        mass_charge = state[:,:,:-4] # if no charge = [m]; with charge = [m, c]
        momentum = state[:,:,-2:] * mass_charge[:,:,0].unsqueeze(2)
        V = torch.cat([state[:,:,-4:-2], momentum], dim=2)
        # Require grad to be able to compute partial derivatives
        if not V.requires_grad:
            V.requires_grad = True
        
        # Compute updated canonical coordinates
        if self.integrator == 'rk4':
            new_canonical_coordinates = self.rk4(dt, mass_charge, V, R_s, R_r)
        elif self.integrator == 'euler':
            new_canonical_coordinates = self.euler(dt, mass_charge, V, R_s, R_r)
        else:
            raise Exception
        
        # Convert back to original state format [x, y, vx, vy]
        velocity = torch.div(new_canonical_coordinates[:,:,2:], mass_charge[:,:,0].unsqueeze(2))
        new_state = torch.cat([new_canonical_coordinates[:,:,:2], velocity], dim=2)
        return new_state
