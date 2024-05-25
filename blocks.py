import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import randint
from graphs import GraphsTuple


# import utils_tf

def broadcast_receiver_nodes_to_edges(graph: GraphsTuple):
    for node in graph.nodes:
        if node.device!="cuda:0" and torch.cuda.is_available():
            node.to("cuda:0")

    for i in graph.receivers:
        if i.device!="cuda:0" and torch.cuda.is_available():
            i.to("cuda:0")
    return graph.nodes.index_select(index=graph.receivers, dim=0)


def broadcast_sender_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.senders.long(), dim=0)


def broadcast_globals_to_edges(graph: GraphsTuple):
    N_edges = graph.edges.shape[0]
    return graph.globals.repeat(N_edges, 1)


def broadcast_globals_to_nodes(graph: GraphsTuple):
    N_nodes = graph.nodes.shape[0]
    return graph.globals.repeat(N_nodes, 1)


class Aggregator(nn.Module):
    def __init__(self, mode):
        super(Aggregator, self).__init__()
        self.mode = mode

    def forward(self, graph):
        edges = graph.edges
        nodes = graph.nodes
        if self.mode == 'receivers':
            indices = graph.receivers
        elif self.mode == 'senders':
            indices = graph.senders
        else:
            raise AttributeError("invalid parameter `mode`")
        N_edges, N_features = edges.shape
        N_nodes = nodes.shape[0]
        aggrated_list = []
        for i in range(N_nodes):
            aggrated = edges[indices == i] #all edges features that have node i as sender/receiver
            if aggrated.shape[0] == 0:
                aggrated = torch.zeros(1, N_features)
            aggrated_list.append(torch.sum(aggrated, dim=0))
        return torch.stack(aggrated_list,dim=0)


class EdgeBlock(nn.Module):
    def __init__(self,
                 nodedim, 
                 edgedim,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        super(EdgeBlock, self).__init__()
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals
        N_features = 0
        
        if self._use_edges:
            N_features += edgedim
        if self._use_receiver_nodes:
            N_features += nodedim
        if self._use_sender_nodes:
            N_features += nodedim
        self.edgefn = nn.Linear(N_features, edgedim)

    def forward(self, graph: GraphsTuple):
        edges_to_collect = []
        #print(graph.edges.shape,graph.nodes.shape)
        if self._use_edges:
            edges_to_collect.append(torch.tensor(graph.edges))  # edge feature  (50,6)

        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))  # (50,5)
            # receiver=(50,) 
            

        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))  # (50,5)
            

        if self._use_globals:
            edges_to_collect.append(broadcast_globals_to_edges(graph))  # (50,)

        collected_edges = torch.cat(edges_to_collect, dim=-1) #torch.cat(torch.tensor(edges_to_collect), dim=1)
        updated_edges = self.edgefn(collected_edges)
        return graph.replace(edges=updated_edges)


class NodeBlock(nn.Module):

    def __init__(self,
                 nodedim, 
                 edgedim,
                 use_received_edges=True,
                 use_sent_edges=False,
                 use_nodes=True,
                 use_globals=False):
        super(NodeBlock, self).__init__()
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        N_features = 0
        
        if self._use_nodes:
            N_features += nodedim
        if self._use_received_edges:
            N_features += edgedim
        if self._use_sent_edges:
            N_features += edgedim
        self.nodefn = nn.Linear(N_features, nodedim)
        self._received_edges_aggregator = Aggregator('receivers')
        self._sent_edges_aggregator = Aggregator('senders')

    def forward(self, graph):

        nodes_to_collect = []
        # nodes: (24,5)
        # edges: (50,10)  
        # global: (4,4)

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))  # (24,10)

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_nodes:
            nodes_to_collect.append(torch.tensor(graph.nodes))

        if self._use_globals:
            nodes_to_collect.append(broadcast_globals_to_nodes(graph))  # (24,4)

        collected_nodes = torch.cat(nodes_to_collect, dim=-1)  
        updated_nodes = self.nodefn(collected_nodes)  

        return graph.replace(nodes=updated_nodes)


class GlobalBlock(nn.Module):
    def __init__(self,
                 use_edges=True,
                 use_nodes=True,
                 use_globals=True):

        super(GlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals


    def forward(self, graph):
        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph))

        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph))

        if self._use_globals:
            globals_to_collect.append(graph.globals)

        collected_globals = torch.cat(globals_to_collect, dim=1)
        updated_globals = self._global_model(collected_globals)

        return graph.replace(globals=updated_globals)