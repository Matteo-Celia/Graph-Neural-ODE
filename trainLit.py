import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.optim as optim
import lightning as L
from math import inf
import time
import argparse
from model import GNSTODE
from train import ReconstructionLoss
from data import TrajectoryDataset_New


class LitGNSTODE(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.model(x)
        criterion = ReconstructionLoss()
        loss = criterion(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    


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


    device = torch.device("cuda" if ((not cpu) and torch.cuda.is_available()) else "cpu")#cuda:0
    model.to(device)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)#3e-4
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay) # decay every 2 * 10^5 with lower imit of 10^-7
    if device == "cuda":
        trainer = L.Trainer(accelerator="cuda", devices=-1)
    else:
        trainer = L.Trainer()
        
    Lit_model = LitGNSTODE(model)
    
    trainer.fit(model=Lit_model, train_dataloaders=train_loader)


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
    parser.add_argument('--validate_every', action="store", type=int, default=5,
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

    

    
