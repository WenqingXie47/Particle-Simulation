import numpy as np
import torch
import random
import os

class NBodyDataset():
    """
    NBodyDataset

    """
    def __init__(self, n_hidden=0, data_folder="data", dataset_name="se3_transformer", partition='train', max_samples=1e8):
        self.n_hidden = n_hidden
        self.max_samples = int(max_samples)
        
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.partition = partition
        if partition == 'val':
            self.partition = 'valid'        
        self.load(
            self.data_folder, 
            self.dataset_name,
            self.partition,
        )
        self.preprocess()
        
        
    def load(self, data_folder, dataset_name, partition):
        
        dir = os.path.join(*[data_folder,dataset_name])
        
        self.loc = np.load(os.path.join(dir, f"loc_{partition}.npy"))
        self.vel = np.load(os.path.join(dir, f"vel_{partition}.npy"))
        self.charges = np.load(os.path.join(dir, f"charges_{partition}.npy"))
        self.masses = np.load(os.path.join(dir, f"masses_{partition}.npy"))
        self.edge_attr = np.load(os.path.join(dir, f"edges_{partition}.npy"))



    def preprocess(self):
        """preprocess data

        Args:
            loc (numpy.ndarray): shape (n_sims, n_iters, n_dims, n_particles)
            vel (numpy.ndarray): shape (n_sims, n_iters, n_dims, n_particles)
            edges (numpy.ndarray): shape (n_sims, n_particles , n_particles)
            charges (numpy.ndarray): shape (n_sims, n_particles)

        Returns:
            _type_: _description_
        """
        
        # swap n_nodes <--> n_features dimensions
        self.loc = self.loc.transpose(2, 3)
        self.vel = self.vel.transpose(2, 3)
        
        #Initialize edges and edge_attributes
        edge_attr = []
        rows, cols = [], []
        n_nodes = self.loc.size(2)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(self.edge_attr[:, i, j])
                    rows.append(i)
                    cols.append(j)
        self.edges = np.array([rows, cols])
        # swap n_nodes <--> batch_size and add nf dimension
        self.edge_attr = edge_attr.transpose(0, 1).unsqueeze(2)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)

        
    def get_n_nodes(self):
        return self.loc.shape[1]

    def __getitem__(self, i):
        loc = self.loc[i]
        vel = self.vel[i]
        charges = self.charges[i]
        masses = self.masses[i]
        edge_attr = self.edge_attr[i]
        
        return loc[:-1], vel[:1], charges, masses, edge_attr, vel[:-1]-vel[1:]


    def __len__(self):
        return len(self.loc)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


    
class NBodySlidingWindowDataset(NBodyDataset):
    
    def __init__(self, lag=5, partition='train', max_samples=100000000, dataset_name="se3_transformer"):
        super().__init__(partition, max_samples, dataset_name)
        self.lag = lag
    
    def __getitem__(self, i):
        
        loc, vel, edge_attr, charges = self.data
        n_sequences = loc.shape[0]
        n_iters = loc.shape[1]
        n_samples_per_seq = n_iters - self.lag
        
        seq_index = i/n_samples_per_seq
        iter_index = i%n_samples_per_seq
        
        
        return (
            loc[seq_index, iter_index:iter_index+self.lag],
            vel[seq_index, iter_index:iter_index+self.lag],
            edge_attr,
            charges,
            vel[seq_index, iter_index+self.lag] - vel[seq_index, iter_index+self.lag-1]
        )
            
    

    def __len__(self):
        loc, vel, edge_attr, charges = self.data
        n_sequences = loc.shape[0]
        n_iters = loc.shape[1]
        n_samples_per_seq = n_iters - self.lag
        return n_sequences*n_samples_per_seq 