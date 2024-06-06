import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from math import inf
import time
import argparse
import random
import string

from data import TrajectoryDataset, TrajectoryDataset_New
from model import  GNSTODE
from utils import reconstruction_loss,PBC_MSE_loss, create_folder,  data_dicts_to_graphs_tuple, pbc_diff, build_GraphTuple
from eval import evaluate_model

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        
        
    def forward(self, predictions, targets):
        
        diff = predictions - targets
        frobenius_norm = torch.linalg.matrix_norm(diff.float()) 
        loss = frobenius_norm**2
        rec_loss = torch.sum(loss)

        return rec_loss



def training_step_dynamic_graph(model, data, dt, device, accumulate_steps, box_size, graph_type, simulation_type):

  
    inputs, targets= data
    targets = targets.squeeze(0)
    
    inputs.requires_grad_(True)
    targets.requires_grad_(True)
    # Push data to the GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    

    # Forward pass (and time it)
    start_time = time.perf_counter_ns()
    outputs = model(inputs, dt=dt)
    end_time = time.perf_counter_ns()
    #print(outputs.requires_grad)
    # Backward
    criterion = ReconstructionLoss()
    loss = criterion(outputs, targets)

    loss = loss / accumulate_steps
    loss.backward()

    return loss.item(), (end_time - start_time)

def validation_step_dynamic_graph(model, test_data, dt, device, box_size, graph_type):

    
    inputs, targets = test_data
    targets = targets.squeeze(0)
    # Push to GPU
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    
    
    # Get outputs
    outputs = model(inputs, dt=dt)
    criterion = ReconstructionLoss()
    test_loss = criterion(outputs, targets)
    #loss = reconstruction_loss(outputs, targets) 
    # Get loss
    #test_loss = reconstruction_loss(outputs, targets)#PBC_MSE_loss(outputs, targets[:,:,-4:], box_size=box_size).cpu().detach()

    return test_loss.item()

def train_model(model_type="GNSTODE", dataset="3_particles_gravity", learning_rate=1e-3, lr_decay=0.97725, batch_size=50, epochs=10, accumulate_steps=1, model_dir="models", data_dir="data",
                hidden_units=-1, validate=True, validate_epochs=2, graph_type='_nn', integrator='dopri5',
                pre_load_graphs=False, data_loader_workers=2, smooth_lr_decay=False, target_step=1, cpu=False, experiment_dir="", log_dir="runs", resume_checkpoint="", save_after_time=0):
    # Track time for saving after x seconds
    start_time_for_save = time.monotonic()

    # Set CPU paralelization to maximum
    torch.set_num_threads(torch.get_num_threads())

    # Load training dataset
    train_set = TrajectoryDataset_New(folder_path=os.path.join(data_dir, dataset), split='train', graph_type=graph_type, pre_load_graphs=pre_load_graphs, target_step=target_step)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=data_loader_workers, pin_memory=True) #collate_fn=collate_into_one_graph,

    # Load validation dataset
    validation_set = TrajectoryDataset_New(folder_path=os.path.join(data_dir, dataset), split='validation', graph_type=graph_type, pre_load_graphs=pre_load_graphs, target_step=target_step)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=data_loader_workers,  pin_memory=True) #collate_fn=collate_into_one_graph,

    # Get parameters form dataset
    box_size = train_set.box_size
    time_step = train_set.time_step
    n_particles = train_set.n_particles
    simulation_type = train_set.simulation_type
    traj_len = train_set.trajectory_len

    
    model = GNSTODE(n_particles, box_size=box_size, integrator=integrator, simulation_type=simulation_type)


    device = torch.device("cuda:0" if ((not cpu) and torch.cuda.is_available()) else "cpu")#cuda:0
    model.to(device)

    load = True
    if load:
        checkpoint = torch.load('load/GNSTODE_euler_lr_0.0003_decay_0.1_epochs_10_batch_size_50_accumulate_steps_1_graph__nn_target_step_1_20240602-134653_91372NHJ0T_checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)#3e-4
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)#3e-4

    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay) # decay every 2 * 10^5 with lower imit of 10^-7

# If needed, load the optimizer state
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)#3e-4
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay) # decay every 2 * 10^5 with lower imit of 10^-7

    # Track iterations and running loss for loging
    n_iter = 0
    running_loss = 0.0

    # if graph_type == 'fully_connected':
    #     # Build graph - R_s one hot encoding of edge sender id, R_r: one hot encoding of reciever id
    #     # All possible n*(n-1) are modeled as present - matrix shape (n-1) x n
    #     # Repeated for each sample in the batch (final size: batch x n(n-1) x n)
    #     R_s, R_r = full_graph_senders_and_recievers(n_particles, batch_size=batch_size, device=device)


    # Build a tensor of step sizes for each sample in the batch
    dt = torch.Tensor([time_step]).to(device).unsqueeze(0) * target_step

    # Use current/start time to identify saved model and log dir
    rand_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    start_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_type}{f'_{integrator}' if integrator else ''}_lr_{learning_rate}_decay_{lr_decay}_epochs_{epochs}_batch_size_{batch_size}_accumulate_steps_{accumulate_steps}{f'_hidden_units_{hidden_units}' if hidden_units > 0 else ''}_graph_{graph_type}_target_step_{target_step}_{start_time}_{rand_string}"
    # if len(resume_checkpoint) > 0:
    #     print('Resuming checkpoint')
    #     model_name = resume_checkpoint.replace('_checkpoint.tar', '')
    #     checkpoint = torch.load(os.path.join(model_dir, experiment_dir, dataset, resume_checkpoint))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     starting_epoch = checkpoint['epoch']
    #     n_iter = starting_epoch * len(train_loader)
    # else:
    #     # Create model filename to use for saved model params and tensorboard log dir
    #     rand_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    #     model_name = f"{model_type}{f'_{integrator}' if integrator_model else ''}_lr_{learning_rate}_decay_{lr_decay}_epochs_{epochs}_batch_size_{batch_size}_accumulate_steps_{accumulate_steps}{f'_hidden_units_{hidden_units}' if hidden_units > 0 else ''}_graph_{graph_type}_target_step_{target_step}_{start_time}_{rand_string}"
    #     starting_epoch = 0
    starting_epoch = 0
    # Setup direcotries for logs and models
    if len(experiment_dir) > 0:
        logdir=os.path.join(log_dir, experiment_dir, dataset, model_name)
        model_save_path = os.path.join(model_dir, experiment_dir, dataset)
    else:
        logdir=os.path.join(log_dir, dataset, model_name)
        model_save_path = os.path.join(model_dir, dataset)
    
    # Setup Writer for Tensorboard
    writer = SummaryWriter(log_dir=logdir)

    # Log lr for tensorboard
    writer.add_scalar('LearningRate/optimizer_lr', optimizer.param_groups[0]['lr'], n_iter)

    # Create output folder for the dataset models
    create_folder(model_save_path)

    # Track forward pass time
    forward_pass_times = []

    print("Training model %s on %s dataset" % (model_name, dataset))
    model.train() #needed?
    for epoch in range(starting_epoch, epochs):
        optimizer.zero_grad()
        print(f"epoch : {epoch}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for i, data in enumerate(pbar):
            # Do one training step and get loss value
            
            loss_value, forward_pass_time = training_step_dynamic_graph(model, data, dt, device, accumulate_steps, box_size, graph_type, simulation_type)
            running_loss += loss_value
            forward_pass_times.append(forward_pass_time)

            # Do an optimizer step on accumulated gradients
            if (i+1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                n_iter += 1

                # Log training loss every 200 optimizer steps
                if n_iter % 200 == 0:
                    writer.add_scalar('Loss/train', running_loss / 200, n_iter)
                    running_loss = 0.0

            # Decay learning rate every 200k steps with lower imit of 10^-7 (if smooth decay is not set)
            if ((n_iter % (2 * 10**5) == 0) and not smooth_lr_decay) and (optimizer.param_groups[0]['lr'] > 10**(-7)):
                scheduler.step()
                # Log lr for tensorboard
                writer.add_scalar('LearningRate/optimizer_lr', optimizer.param_groups[0]['lr'], n_iter)

            # Save model after specified number of seconds (if set)
            if (save_after_time > 0):
                if ((time.monotonic() - start_time_for_save) > save_after_time):
                    torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
                    start_time_for_save = time.monotonic()
        
        # Decay learning rate every epoch with lower imit of 10^-7 (if smooth decay is set)
        if (smooth_lr_decay) and (optimizer.param_groups[0]['lr'] > 10**(-7)):
            scheduler.step()
            # Log lr for tensorboard
            writer.add_scalar('LearningRate/optimizer_lr', optimizer.param_groups[0]['lr'], n_iter)

        # Save model each epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, model_name))

        # Save a checkpoint each epoch
        torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()},os.path.join(model_save_path, model_name + "_checkpoint.tar"))
            
        # Evaluate on validation set every validate_epochs epoch. Always validate and save the model after the last epoch
        # No with torch.no_grad() since HOGN needs gradients
        if (validate and ((epoch+1) % validate_epochs == 0)) or (epoch == epochs - 1) :
            print("Validation starting")
            model.eval()
            running_test_loss = 0.0
            for p in model.parameters():
                p.require_grads = False
            pbarval = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for j, test_data in enumerate(pbarval):
                # Ensure no grad is left before validation step
                optimizer.zero_grad()
                #if model_type == "NewMultiLevelHOGNDown5":
                #    torch.cuda.empty_cache()
                # Do a validation step
                # if graph_type == 'fully_connected':
                #     loss_value =  validation_step_static_graph(model, test_data, R_s, R_r, dt, device, box_size)
                # else:
                loss_value = validation_step_dynamic_graph(model, test_data, dt, device, box_size, graph_type)
                running_test_loss += loss_value
            model.train()
            for p in model.parameters():
                p.require_grads = True

            # Log validation loss for tensorboard
            validation_loss = running_test_loss / len(validation_loader)
            print(f"Validation loss at epoch {epoch} is : {validation_loss}")
            writer.add_scalar('Loss/validation', validation_loss, n_iter)


    writer.close()

    # Print validation loss at the end of the training
    print(f'Finished Training. Final validation loss is {validation_loss}')
    forward_pass_times = (np.array(forward_pass_times) / 1000000) # in ms instead of ns
    print(f'Forward step took {np.mean(forward_pass_times)} ms on average (std: {np.std(forward_pass_times)})')
    # Return the filename of the model
    return model_name


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', action='store', default="models",
                        dest='model_dir',
                        help='Set model directory')
    parser.add_argument('--data_dir', action='store', default="data",
                        dest='data_dir',
                        help='Set data directory')
    parser.add_argument('--model', action='store', default="GNSTODE",
                        dest='model',
                        help='Set model type to train')
    parser.add_argument('--lr', action='store', type=float, default=0.0003,
                        dest='lr',
                        help='Set learning rate')
    parser.add_argument('--lr_decay', action='store', type=float, default=0.1, # use 0.97725 instead if decaying every epoch (smooth_lr_decay flag)
                        dest='lr_decay',
                        help='Set learning rate decay')
    parser.add_argument('--batch_size', action='store', type=int, default=50, #50
                        dest='batch_size',
                        help='Set batch size')
    parser.add_argument('--epochs', action='store', type=int, default=10, #200
                        dest='epochs',
                        help='Set number of epochs')
    parser.add_argument('--accumulate_steps', action='store', type=int, default=1,
                        dest='accumulate_steps',
                        help='Set number of epochs')
    parser.add_argument('--dataset', action='store', default="3_particles_gravity",
                        dest='dataset',
                        help='Set dataset to use')
    parser.add_argument('--hidden_units', action="store", type=int, default=-1,
                        dest='hidden_units',
                        help='Set number of hidden units linear layers of MLPs will use')
    parser.add_argument('--dont_validate', action="store_false", default=True,
                        dest='validate',
                        help='Do not validate model each epoch')
    parser.add_argument('--validate_every', action="store", type=int, default=1,
                        dest='validate_every',
                        help='Validate model every n epochs')
    parser.add_argument('--graph_type', action='store', default="_nn",
                        dest='graph_type',
                        help='Set type of the graaph to use')
    parser.add_argument('--integrator', action='store', default="euler", #dopri5
                        dest='integrator',
                        help='Set integrator to use for HOGN and OGN models')
    parser.add_argument('--dont_pre_load_graphs', action="store_false", default=True,
                        dest='pre_load_graphs',
                        help='Do not pre load graphs into memory (for the Dataset object). Use this flag if there is not enough RAM')
    parser.add_argument('--data_loader_workers', action="store", type=int, default=2,
                        dest='data_loader_workers',
                        help='Number of dataloader workers to use')
    parser.add_argument('--smooth_lr_decay', action="store_true", default=False,
                        dest='smooth_lr_decay',
                        help='Decay LR every epoch instead of every 200k training steps')
    parser.add_argument('--target_step', action="store", type=int, default=1,
                        dest='target_step',
                        help='How many steps into the future target will be')
    parser.add_argument('--cpu', action="store_true", default=False,
                        dest='cpu',
                        help='Train model on CPU (slow, not tested properly)')
    parser.add_argument('--experiment_dir', action='store', default="",
                        dest='experiment_dir',
                        help='Set experiment sub-directory')
    parser.add_argument('--log_dir', action='store', default="runs",
                        dest='log_dir',
                        help='Set directory for tensorboard logs')
    parser.add_argument('--resume_checkpoint', action='store', default="",
                        dest='resume_checkpoint',
                        help='Load the specified checkpoint to resume training')
    parser.add_argument('--save_after_time', action="store", type=int, default=0,
                        dest='save_after_time',
                        help='Save model after x seconds since start')
    parser.add_argument('--eval', action="store_true", default=False,
                        dest='eval',
                        help='Evaluate the trained model on test set')                
    arguments = parser.parse_args()

    # Run training using parsed arguments
    model_file_name = train_model(model_type=arguments.model, dataset=arguments.dataset, learning_rate=arguments.lr, lr_decay=arguments.lr_decay, batch_size=arguments.batch_size,
                                epochs=arguments.epochs, accumulate_steps=arguments.accumulate_steps, model_dir=arguments.model_dir, data_dir=arguments.data_dir,
                                hidden_units=arguments.hidden_units, validate=arguments.validate, validate_epochs=arguments.validate_every, graph_type=arguments.graph_type,
                                integrator=arguments.integrator, pre_load_graphs=arguments.pre_load_graphs, data_loader_workers=arguments.data_loader_workers, 
                                smooth_lr_decay=arguments.smooth_lr_decay, target_step=arguments.target_step, cpu=arguments.cpu, experiment_dir=arguments.experiment_dir, 
                                log_dir=arguments.log_dir, resume_checkpoint=arguments.resume_checkpoint, save_after_time=arguments.save_after_time)

    if arguments.eval:
        evaluate_model(model_file=model_file_name, dataset=arguments.dataset, model_dir=arguments.model_dir, data_dir=arguments.data_dir, experiment_dir=arguments.experiment_dir, pre_load_graphs=arguments.pre_load_graphs)
    