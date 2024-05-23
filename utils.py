import torch
import torch.nn as nn
import blocks
import torch.nn.functional as F
import numpy as np
import os
import shutil
from itertools import accumulate, chain
from collections import abc
from random import randint
from graphs import GraphsTuple

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

def split_matrix_np(matrix, num_chunks, chunk_size, flatten=True):
    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    
    
    if flatten:
    # Split the matrix
        submatrices = [matrix[i * chunk_size:(i + 1) * chunk_size].reshape(-1) for i in range(num_chunks)]
    else:
        submatrices = [matrix[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    
    return submatrices

def build_GraphTuple(inputs, R_s, R_r):

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
            distances = np.linalg.norm(pbc_diff(inputs[i, indices, -4:-2], inputs[i, j, -4:-2][np.newaxis, :], box_size=6), axis=-1)
            dist_list.append(np.array(distances))
        
        edges = [dist_list[R_s[i].astype(np.int64)][R_r[i].astype(np.int64)] for i in range(len(R_s))]
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

def build_senders_receivers(trajectory, neighbour_count =2, box_size=6): #15
    
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