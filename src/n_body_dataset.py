import numpy as np
import torch
import random
import os

class NBodyDataset():
    """
    NBodyDataset

    """
    def __init__(self, n_hidden=0, dataset_name="se3_transformer", partition='train', max_samples=1e8):
        self.n_hidden = n_hidden
        self.max_samples = int(max_samples)
        
        self.dataset_name = dataset_name
        self.partition = partition
        if partition == 'val':
            self.partition = 'valid'
        

        self.dataset_name = dataset_name        
        self.suffix = "_" + self.partition + "_" + self.dataset_name

        
        loc, vel, charges, masses, edges = NBodyDataset.load(dir="n_body_system/dataset",suffix=self.suffix)
        self.loc, self.vel, self.charges, self.edges, self.edge_attr = NBodyDataset.preprocess(
            loc, vel, edges, charges
        )

    def load(data_folder, dataset_name, partition):
        
        dir = os.path.join(*[data_folder,dataset_name])
        loc = np.load(os.path.join(dir, f"loc_{partition}.npy"))
        vel = np.load(os.path.join(dir, f"vel_{partition}.npy"))
        charges = np.load(os.path.join(dir, f"charges_{partition}.npy"))
        masses = np.load(os.path.join(dir, f"masses_{partition}.npy"))
        edges = np.load(os.path.join(dir, f"edges_{partition}.npy"))
        return loc, vel, charges, masses, edges



    def preprocess(loc, vel, charges, edges):
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
        loc, vel = loc.transpose(2, 3), vel.transpose(2, 3)
        
        #Initialize edges and edge_attributes
        edge_attr = []
        rows, cols = [], []
        n_nodes = loc.size(2)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = np.array([rows, cols])
        edge_attr = edge_attr.transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension

        return loc, vel, charges, edges, edge_attr


    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
        
        
    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]


        return loc[:-1], vel[:1], edge_attr, charges, vel[:-1]-vel[1:]


    def __len__(self):
        return len(self.data[0])

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