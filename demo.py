import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
#import model
import utils_np

def get_graph_data_dicts(num_nodes, num_edges):
    GLOBAL_SIZE = 4
    NODE_SIZE = 5
    EDGE_SIZE = 6

    data_dict1 = {
      "globals": np.random.rand(GLOBAL_SIZE).astype(np.float32),
      "nodes": np.random.rand(num_nodes, NODE_SIZE).astype(np.float32),
      "edges": np.random.rand(num_edges, EDGE_SIZE).astype(np.float32),
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
    }

    data_dict2 = {
      "globals": np.random.rand(GLOBAL_SIZE).astype(np.float32),
      "nodes": np.random.rand(num_nodes, NODE_SIZE).astype(np.float32),
      "edges": np.random.rand(num_edges, EDGE_SIZE).astype(np.float32),
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
    }
    return [data_dict1,data_dict2]

if __name__ == '__main__':
    
    graph_dicts = get_graph_data_dicts(num_nodes=9, num_edges=25)
    input_graphs = utils_np.data_dicts_to_graphs_tuple(graph_dicts)
    print("graph dicts:")
    print(graph_dicts)
    print('input_graphs')
    print(input_graphs)

    # graph_network = modules.GraphNetwork(input_graphs)

    # output_graphs = graph_network(input_graphs)

    # print('output_graphs')
    # print(output_graphs)