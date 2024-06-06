import torch
import numpy as np
import os
import shutil
import argparse
import re

from data import TrajectoryDataset, TrajectoryDataset_New
from model import GNSTODE
from utils import pbc_rms_error, pbc_mean_relative_energy_error, recreate_folder, create_folder

class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        
        
    def forward(self, predictions, targets, ):
        
        diff = predictions - targets
        frobenius_norm = torch.linalg.matrix_norm(diff.float(), p='fro') 
        sqrd_fn = frobenius_norm**2
        rec_loss = torch.sum(sqrd_fn)
        if len(predictions.shape) ==4:
            den = 1/(2*predictions.shape[-2]*predictions.shape[-3]*predictions.shape[-4])
        else:
            den = 1/(2*predictions.shape[-2]*predictions.shape[-3])
        
        loss = torch.sqrt(den*rec_loss)

        return loss


def evaluate_model(model_file="", dataset="3_particles_gravity", model_dataset="", graph_type="", model_dir="models", data_dir="data", experiment_dir="", pre_load_graphs=True, start_id=0, end_id=-1):

    # Set evaluation dataset as model dataset (dataset model was trained on) if no model dataset was specified
    if len(model_dataset) < 1:
        model_dataset = dataset

    # Model path and output folder path
    if len(experiment_dir) > 0:
        model_path = os.path.join(model_dir, experiment_dir, model_dataset, model_file)
        output_folder_path = os.path.join(data_dir, experiment_dir, dataset, 'test_predictions', model_file)
    else:
        model_path = os.path.join(model_dir, model_dataset, model_file)
        output_folder_path = os.path.join(data_dir, dataset, 'test_predictions', model_file)

    # Get model type from filename
    model_type = model_file.split('_')[0]

    # Get graph type from the filename if not set
    # if len(graph_type) < 1:
    #     graph_type_search = re.search(r'graph_(\d+(?:_nn|_level_hierarchical))_', model_file)
    #     if graph_type_search:
    #         graph_type = graph_type_search.group(1)
    #     else:
    #         # If graph type is not found set the graph to fully connected
    #         graph_type = '_nn'
    
    # Extract graph type specific params
    # if '_nn' in graph_type:
    #     edges_per_node = int(graph_type.split('_')[0])
    # elif '_level_hierarchical' in graph_type:
    #     hierarchy_levels = int(graph_type.split('_')[0])

    # Get hidden unit count from the filename
    hidden_units_search = re.search(r'hidden_units_(\d+)_', model_file)
    if hidden_units_search:
        hidden_units = int(hidden_units_search.group(1))
    else:
        # Use defaults
        hidden_units = -1

    # Get target_step from the filename
    target_step_search = re.search(r'target_step_(\d+)_', model_file)
    if target_step_search:
        target_step = int(target_step_search.group(1))
    else:
        # Use defaults
        target_step = 1

    # Get integrator type from the filename
    integrator = 'euler' 
    

    # Use batch_size=1 for inference
    batch_size = 1
    
    # Load test data set
    test_set = TrajectoryDataset_New(folder_path=os.path.join(data_dir, dataset), split='test', rollout=True, graph_type=graph_type, target_step=target_step)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Get parameters form dataset
    box_size = test_set.box_size
    time_step = test_set.time_step
    n_particles = test_set.n_particles
    physical_const = test_set.physical_const
    softening = test_set.softening
    softening_radius = test_set.softening_radius
    simulation_type = test_set.simulation_type

    # if model_type == "DeltaGN":
    #     model = DeltaGN(box_size=box_size, edge_output_dim=hidden_units, node_output_dim=hidden_units, simulation_type=simulation_type)
    model = GNSTODE(n_particles, box_size=box_size, integrator=integrator, simulation_type=simulation_type)

    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model_state_dict']) # model path

    # Remove old output subfolder if it exists in case all trajectories will be built
    # if start_id == 0 and end_id == -1:
    #     recreate_folder(output_folder_path)
    # else:
    #     create_folder(output_folder_path)

    # Set proper end_id
    if end_id == -1:
        end_id = len(test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    #if graph_type == 'fully_connected':
        # Build graph - R_s one hot encoding of edge sender id, R_r: one hot encoding of reciever id
        # All possible n*(n-1) are modeled as present - matrix shape (n-1) x n
        # Repeated for each sample in the batch (final size: batch x n(n-1) x n)
     #   R_s, R_r = full_graph_senders_and_recievers(n_particles, batch_size=batch_size, device=device)

    # Build a tensor of step sizes for each sample in the batch
    dt = torch.Tensor([time_step]).to(device).unsqueeze(0).expand(batch_size, 1) * target_step

    # Log all trajectories for RMS over all trajectories
    predicted_trajectories = []

    print("Evaluationg model %s on %s test dataset" % (model_file, dataset))

    for i, data in enumerate(test_loader, 0):
        if start_id <= i < end_id:
            # Get the inputs as full trajectory
            #if graph_type == 'fully_connected':
            inputs, targets = data
            
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # elif '_level_hierarchical' in graph_type:
            #     inputs, targets, R_s, R_r, assignment, V_super, super_graph = data

            #     R_s = R_s.to(device, non_blocking=True)
            #     R_r = R_r.to(device, non_blocking=True)
            #     assignment = [el.to(device, non_blocking=True) for el in assignment]
            #     V_super = [el.to(device, non_blocking=True) for el in V_super]
            #     super_graph = [el.to(device, non_blocking=True) for el in super_graph]
            # elif '_nn' in graph_type:
            #     inputs, targets, R_s, R_r = data
            #     R_s = R_s.to(device, non_blocking=True)
            #     R_r = R_r.to(device, non_blocking=True)
            # else:
            #     raise ValueError('Graph type not recognized')

            # Log the predicted trajecotry
            print(inputs.shape)
            output_trajectory = torch.zeros((inputs.shape[0]+1, inputs.shape[1], inputs.shape[2]))
            output_trajectory[0] = inputs[0] #.numpy()

            # Forward pass 
            current_state = inputs[0].unsqueeze(0)
            current_state = current_state.to(device)
            for j in range(inputs.shape[0]):

                output = model(current_state, dt)
                current_state = output.detach() # Detach to stop graph unroll in next loop iteration 
                
                output_trajectory[j+1, :, :] = current_state.detach() # [timesteps, particles, state]; state = [m,x,y,v_x,v_y]

            rmse = RMSE()
            print("RMSE for trajectory %i: %f" % (i , rmse(output_trajectory, targets))) 
        
            # Save the predicted trajectory
            output_filename = os.path.join(output_folder_path,"predicted_trajectory_{i}.npy".format(i=i))
            np.save(output_filename, output_trajectory)

            # Log for RMS over all trajectories
            predicted_trajectories.append(output_trajectory)


    # RMS over all trajectories
    predicted_trajectories = np.stack(predicted_trajectories, axis=0) # [trajectories, timesteps, particles, state]; state = [m,x,y,v_x,v_y]
    print("RMSE over all trajectories: %f" % (rmse(predicted_trajectories, test_set.trajectories[start_id:end_id,target_step::target_step,:,:]))) #[:,1:,:,1:]
    #print("Mean relative energy error over all trajectories: %f" % (pbc_mean_relative_energy_error(predicted_trajectories, box_size=box_size, physical_const=physical_const, softening=softening, softening_radius=softening_radius)))

    print('Finished evaluation')

    # Return the output path
    return output_folder_path


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()              #models
    parser.add_argument('--model_dir', action='store', default="models",
                        dest='model_dir',
                        help='Set model directory')
    parser.add_argument('--data_dir', action='store', default="data",
                        dest='data_dir',
                        help='Set data directory')
    parser.add_argument('--model_file', action='store', default="",
                        dest='model_file',
                        help='Set model parameter file to use for evaluation')
    parser.add_argument('--model_dataset', action='store', default="",
                        dest='model_dataset',
                        help='Set dataset model was trained on (by default --dataset value is used)')
    parser.add_argument('--dataset', action='store', default="3_particles_gravity",
                        dest='dataset',
                        help='Set dataset to use (if model_dataset is set this dataset is used for evaluation only)')
    parser.add_argument('--graph_type', action='store', default="",
                        dest='graph_type',
                        help='Set type of the graph to use')
    parser.add_argument('--experiment_dir', action='store', default="",
                        dest='experiment_dir',
                        help='Set experiment sub-directory')
    parser.add_argument('--dont_pre_load_graphs', action="store_false", default=True,
                        dest='pre_load_graphs',
                        help='Do not pre load graphs into memory (for the Dataset object). Use this flag if there is not enough RAM')
    parser.add_argument('--start_id', action='store', type=int, default=0,
                        dest='start_id',
                        help='Set start id of trajectory range to evaluate')
    parser.add_argument('--end_id', action='store', type=int, default=-1,
                        dest='end_id',
                        help='Set end id of trajectory range to evaluate')
    arguments = parser.parse_args()
    # Evaluate model using parsed arguments
    evaluate_model(model_file=arguments.model_file, dataset=arguments.dataset, model_dataset=arguments.model_dataset, graph_type=arguments.graph_type, model_dir=arguments.model_dir,
                    data_dir=arguments.data_dir, experiment_dir=arguments.experiment_dir, pre_load_graphs=arguments.pre_load_graphs, start_id=arguments.start_id, end_id=arguments.end_id)