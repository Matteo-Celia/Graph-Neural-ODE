import torch
import torch.nn as nn
import blocks
import torch.nn.functional as F
import numpy as np
import os
import shutil
from itertools import accumulate, chain
from scipy.spatial.distance import cdist
from collections import abc
from random import randint
import graphs
import collections

NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

GRAPH_DATA_FIELDS = graphs.GRAPH_DATA_FIELDS
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS
ALL_FIELDS = graphs.ALL_FIELDS

GRAPH_NX_FEATURES_KEY = "features"


def _check_valid_keys(keys):
  if any([x in keys for x in [EDGES, RECEIVERS, SENDERS]]):
    if not (RECEIVERS in keys and SENDERS in keys):
      raise ValueError("If edges are present, senders and receivers should "
                       "both be defined.")


def _defined_keys(dict_):
  return {k for k, v in dict_.items() if v is not None}

def recreate_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError:
        print("Directory %s does not exist" % folder_path)
    else:
        print("Successfully deleted the old directory %s" % folder_path)
    try:
        os.makedirs(folder_path)
    except OSError:
        print("Creation of the directory %s failed" % folder_path)
    else:
        print("Successfully created the directory %s" % folder_path)

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        print("Directory %s exists" % folder_path)
    else:
        print("Successfully created the directory %s" % folder_path)


def _check_valid_sets_of_keys(dicts):
  """Checks that all dictionaries have exactly the same valid key sets."""
  prev_keys = None
  for dict_ in dicts:
    current_keys = _defined_keys(dict_)
    _check_valid_keys(current_keys)
    if prev_keys and current_keys != prev_keys:
      raise ValueError(
          "Different set of keys found when iterating over data dictionaries "
          "({} vs {})".format(prev_keys, current_keys))
    prev_keys = current_keys

def _to_compatible_data_dicts(data_dicts):
  """Converts the content of `data_dicts` to arrays of the right type.

  All fields are converted to numpy arrays. The index fields (`SENDERS` and
  `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `np.int32`.

  Args:
    data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and values
      either `None`s, or quantities that can be converted to numpy arrays.

  Returns:
    A list of dictionaries containing numpy arrays or `None`s.
  """
  results = []
  for data_dict in data_dicts:
    result = {}
    for k, v in data_dict.items():
      if v is None:
        result[k] = None
      else:
        dtype = np.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else None
        result[k] = np.asarray(v, dtype)
    results.append(result)
  return results

def _compute_stacked_offsets(sizes, repeats):
  """Computes offsets to add to indices of stacked np arrays.

  When a set of np arrays are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked np array. This
  computes those offsets.

  Args:
    sizes: A 1D sequence of np arrays of the sizes per graph.
    repeats: A 1D sequence of np arrays of the number of repeats per graph.

  Returns:
    The index offset per graph.
  """
  return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.

  The N_NODE field is filled if the graph contains a non-None NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-None RECEIVERS field;
  otherwise, it is set to 0.

  Args:
    data_dict: An input `dict`.

  Returns:
    The data `dict` with number fields.
  """
  dct = data_dict.copy()
  for number_field, data_field in [[N_NODE, NODES], [N_EDGE, RECEIVERS]]:
    if dct.get(number_field) is None:
      if dct[data_field] is not None:
        dct[number_field] = np.array(
            np.shape(dct[data_field])[0], dtype=np.int32)
      else:
        dct[number_field] = np.array(0, dtype=np.int32)
  return dct

def _concatenate_data_dicts(data_dicts):
  """Concatenate a list of data dicts to create the equivalent batched graph.

  Args:
    data_dicts: An iterable of data dictionaries with keys `GRAPH_DATA_FIELDS`,
      plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`. Each dictionary is
      representing a single graph.

  Returns:
    A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
    representing the concatenated graphs.
  """
  # Create a single dict with fields that contain sequences of graph tensors.
  concatenated_dicts = collections.defaultdict(lambda: [])
  for data_dict in data_dicts:
    data_dict = _populate_number_fields(data_dict)
    for k, v in data_dict.items():
      if v is not None:
        concatenated_dicts[k].append(v)
      else:
        concatenated_dicts[k] = None

  concatenated_dicts = dict(concatenated_dicts)

  for field, arrays in concatenated_dicts.items():
    if arrays is None:
      concatenated_dicts[field] = None
    elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
      concatenated_dicts[field] = np.stack(arrays)
    else:
      concatenated_dicts[field] = np.concatenate(arrays, axis=0)

  if concatenated_dicts[RECEIVERS] is not None:
    offset = _compute_stacked_offsets(concatenated_dicts[N_NODE],
                                      concatenated_dicts[N_EDGE])
    for field in (RECEIVERS, SENDERS):
      concatenated_dicts[field] += offset

  return concatenated_dicts

def data_dicts_to_graphs_tuple(data_dicts):
    # for k,v in graph_dicts.items():
    #     graph_dicts[k]=torch.tensor(v)
    # return GraphsTuple(**graph_dicts)
    data_dicts = [dict(d) for d in data_dicts]
    for key in graphs.GRAPH_DATA_FIELDS:
        for data_dict in data_dicts:
            data_dict.setdefault(key, None)
    _check_valid_sets_of_keys(data_dicts)
    data_dicts = _to_compatible_data_dicts(data_dicts)
    return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))

def pbc_diff(pos1, pos2, box_size=6):
    diff = pos1 - pos2
    # Periodic boundry conditions
    diff[diff > box_size/2] = diff[diff > box_size/2] - box_size 
    diff[diff <= -box_size/2] = diff[diff <= -box_size/2] + box_size
    return diff

def split_matrix_np(matrix, num_chunks, chunk_size, flatten=True):
    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    
    
    if flatten:
    # Split the matrix
        submatrices = [matrix[i * chunk_size:(i + 1) * chunk_size].reshape(-1) for i in range(num_chunks)]
    else:
        submatrices = [matrix[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    return submatrices

def build_GraphTuple_old(inputs, R_s, R_r):

    data_dict_list = []

    # compute edge features as distances between nodes
    indices = np.arange(0, inputs.shape[1])
    
    # if simulation_type == "coulomb":
    #     feat_idx = list([0,1,4,5]) # (mass, charge, vx, vy)
    # else:
    #     feat_idx = list([0,3,4]) # (mass, vx, vy) 
    
    for i in range(inputs.shape[0]):

        for j in range(len(inputs[i])):
            dist_list = []
            #distances = np.linalg.norm(pbc_diff(inputs[i, indices, -4:-2], inputs[i, j, -4:-2][np.newaxis, :], box_size=6), axis=-1)
            distances = cdist(inputs[i], inputs[i], 'euclidean')  # Shape (N, N)
            dist_list.append(np.array(distances))
        dist_list = np.squeeze(np.array(dist_list), axis=0)
        edges = [dist_list[R_s[i].astype(np.int64)][R_r[i].astype(np.int64)] for i in range(len(R_s))]
        data_dict= {
        "globals": None,
        "nodes": inputs[i],
        "edges": np.array(edges).reshape((R_s)),
        "senders": R_s[i],
        "receivers": R_r[i]
        }

        data_dict_list.append(dict(data_dict))
    graphs = data_dicts_to_graphs_tuple(data_dict_list)
    return graphs

def build_GraphTuple(inputs, distances, R_s, R_r):

    data_dict_list = []

    for i in range(inputs.shape[0]):
        edge_feat = []
        for j in range(R_s.shape[1]):
            dist = distances[i,R_s[i,j],R_r[i,j]]
            edge_feat.append(dist)
           
        edges = np.array(edge_feat).reshape((R_s.shape[1],1))
        data_dict= {
        "globals": None,
        "nodes": inputs[i],
        "edges": edges,
        "senders": R_s[i],
        "receivers": R_r[i]
        }

        data_dict_list.append(dict(data_dict))

    graphs = data_dicts_to_graphs_tuple(data_dict_list)
    return graphs

def build_senders_receivers_old(trajectory, neighbour_count =2, box_size=6): #15
    
    trajectory_len = trajectory.shape[0]
    n_particles = trajectory.shape[1]
    n_edges = n_particles*neighbour_count
    # Store graph as an edge list (sender, reciever)
    graph = np.zeros([trajectory_len, n_edges, 2])
    
    # Get closest neighbours for each node
    indices = list(range(1, n_particles))
    # Iterate over recievers
    for i in range(n_particles):

        distances = np.linalg.norm(pbc_diff(trajectory[:, indices, -4:-2], trajectory[:, i, -4:-2][:, np.newaxis, :], box_size=box_size), axis=-1)

        if i < n_particles-1:
            indices[i] -= 1
        
        closest_neighbours = np.transpose(np.stack([np.argpartition(distances, neighbour_count-1, axis=-1)[:, :neighbour_count], np.full((trajectory_len, neighbour_count), i)], axis=1), axes=(0,2,1))
        # Fix the sender id (we dropped self connection on reciever) for senders that have true id > reciever id (as currently their id is lower by 1)
        closest_neighbours[:, :, 0][closest_neighbours[:, :, 0] >= closest_neighbours[:, :, 1]] += 1

        graph[:, i*neighbour_count:(i+1)*neighbour_count, :] = closest_neighbours
    
    #return senders , receivers
    return graph[:,:,0], graph[:,:,1]

def build_senders_receivers(inputs, neighbour_count =2, box_size=6): #15
    
    pairwise_distances = torch.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[1]))
    for i in range(inputs.shape[0]):

        distances = cdist(inputs[i], inputs[i], 'euclidean')  # Shape (N, N)
        pairwise_distances[i] = torch.from_numpy(distances)
            
    #print(pairwise_distances.shape,pairwise_distances)

    T, N, _ = pairwise_distances.shape
    k = neighbour_count  # number of nearest neighbors

    # Initialize senders and receivers arrays
    senders = torch.zeros((T,N*k))
    receivers = torch.zeros((T,N*k))

    for t in range(T):
        l=0
        for i in range(N):
            
            # Get distances from node i to all other nodes
            dist = pairwise_distances[t, i]
            
            # Get the indices of the k smallest distances (excluding the node itself)
            nearest_indices = torch.topk(dist, k + 1, largest=False).indices[1:]  # Exclude self (distance 0)
            print(f"nearest indices for node {i} in traj {t} : {nearest_indices}")
            # Append sender and receiver pairs
            for j in nearest_indices:
                senders[t,l]= i
                receivers[t,l]=j.item()
                l+=1

    return pairwise_distances, senders.long(), receivers.long()

# Ensure x and y stay inside the box and follow PBC
def apply_PBC_to_coordinates(coordinates, box_size=6):
    # Only apply to coordinate columns
    coordinates[:,:,-4:-2][coordinates[:,:,-4:-2] >= box_size/2] -= box_size
    coordinates[:,:,-4:-2][coordinates[:,:,-4:-2] < -box_size/2] += box_size
    return coordinates

def apply_PBC_to_distances(distances, box_size=6):
    # Only apply to postion columns
    distances[:,:,-4:-2][distances[:,:,-4:-2] > box_size/2] -= box_size
    distances[:,:,-4:-2][distances[:,:,-4:-2] <= -box_size/2] += box_size
    return distances

# Custom MSE loss that takes periodic boundry conditions into account
def PBC_MSE_loss(output, target, box_size=6):
    # Get difference
    error = output - target
    # Deal with periodic boundry conditions
    error = apply_PBC_to_distances(error, box_size=box_size)
    # Get MSE
    loss = torch.mean((error)**2)
    return loss

def reconstruction_loss(predictions, targets):
    rec_loss = []
    for i in range(predictions.shape[0]):

        diff = predictions[i] - targets[i]
        frobenius_norm = np.linalg.norm(diff, ord='fro')
        rec_loss.append(frobenius_norm**2)
    
    rec_loss = np.array(rec_loss)
    loss = np.sum(rec_loss)
    return loss

def pbc_rms_error(predictions, targets, box_size=6):
    loss = np.sqrt(np.mean(pbc_diff(predictions, targets, box_size=box_size)**2))
    return loss

def pbc_mean_relative_energy_error(predictions, box_size=6, physical_const=2, softening=False, softening_radius=0.001):
    # Mean relative error between first time step and end of trajectory
    total_energy_diff = 0
    for trajectory in predictions:
        true_energy = hamiltonian(trajectory[0], physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
        final_energy = hamiltonian(trajectory[-1], physical_const=physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)
        energy_diff = np.absolute((final_energy - true_energy)/true_energy)
        total_energy_diff += energy_diff
    loss = total_energy_diff / predictions.shape[0]
    return loss

def total_linear_momentum(states):
    masses = states[:, 0]
    velocities = states[:, -2:]

    return np.sum(np.tile(masses, (2, 1)).T * velocities, axis=0)

def total_angular_momentum(states):
    masses = states[:, 0]
    positions = states[:, -4:-2]
    velocities = states[:, -2:]

    return np.sum(np.cross(positions, np.tile(masses, (2, 1)).T * velocities))

def kinetic_energy(states):
    momentum = np.multiply(states[:, -2:], states[:, 0][:, np.newaxis])

def hamiltonian(states, physical_const=2, box_size=6, softening=False, softening_radius=0.001):
    # Total energy of the system
    return kinetic_energy(states) + potential_energy(states, physical_const, box_size=box_size, softening=softening, softening_radius=softening_radius)

def potential_energy(states, physical_const=2, box_size=6, softening=False, softening_radius=0.001):
    n_particles = states.shape[0]

    has_charge = (states.shape[1] == 6) # check if data is from Coulomb simulation

    if n_particles > 1000:
        # If we have many particles iteration is faster
        potential_energy = np.zeros(1)

        for i in range(n_particles):
            diff = pbc_diff(states[:, -4:-2], states[i, -4:-2], box_size=box_size)
            diff[i] = np.zeros((2)) # make sure current particle is not attracted to itself
            distance = np.linalg.norm(diff, axis=1)
            distance[i] = 1 # avoid division by 0 error
            if has_charge:
                m2 =  - np.multiply(states[:, 1], states[i, 1]) # q * q[i]
            else:
                m2 = np.multiply(states[:, 0], states[i, 0]) # m * m[i]
            m2[i] = np.zeros((1)) # make sure current particle is not attracted to itself
            if softening:
                potential_energy -= np.sum((np.divide((physical_const * m2), np.sqrt(distance**2 + softening_radius**2))))
            else:
                potential_energy -= np.sum((np.divide((physical_const * m2), distance)))
        self_ids = np.arange(n_particles)

        return potential_energy/2
    else:
        self_ids = np.arange(n_particles)

        diff = pbc_diff(np.repeat(states[np.newaxis, :, -4:-2], n_particles, axis=0), states[:, np.newaxis, -4:-2], box_size=box_size)
        distance = np.linalg.norm(diff, axis=2)
        distance[self_ids, self_ids] = 1 # avoid division by 0 error
        if has_charge:
            m2 = - np.multiply(np.repeat(states[np.newaxis, :, 1], n_particles, axis=0), states[:, np.newaxis, 1]) # q * q[i]
        else:
            m2 = np.multiply(np.repeat(states[np.newaxis, :, 0], n_particles, axis=0), states[:, np.newaxis, 0]) # m * m[i]
        m2[self_ids, self_ids] = np.zeros((1)) # make sure particle is not attracted to itself
        if softening:
            potential_energy = -np.sum((np.divide((physical_const * m2), np.sqrt(distance**2 + softening_radius**2))))
        else:
            potential_energy = -np.sum((np.divide((physical_const * m2), distance)))

    return potential_energy/2 # we double count