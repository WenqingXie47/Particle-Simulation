
from .particle_simulation import ParticleSimulation
import numpy as np

class GravityParticleSimulation(ParticleSimulation):
    
    def __init__(self, dim=3, box_size=5, n_particles=5, G=1):
        super().__init__(dim, box_size, n_particles)
        self.G=G
        
        
    def randomly_init_particles(self,x_std=0.5,v_std=0.5,m_std=0.5):
        super().init_particles(x_std,v_std)
        self.m = np.random.randn(self.n_particles) * m_std

    
    def init_particles(self,x,v,m):
        self.x = x
        self.v = v
        self.m = m
        
        
    def get_edges(self):
        edges =  np.outer(self.m,self.m)
        np.fill_diagonal(edges, 0)
        return edges
        

    def _particle_interaction_potential_enegry(x,m,interaction_strength):
        U=0
        for i in range(x.shape[1]):
            for j in range(i):
                r = x[:, i] -x[:, j]
                r_norm = np.sqrt((r ** 2).sum())
                U -= interaction_strength * m[i] * m[j] / r_norm
        return U
    
 
    def _potential_energy(self, x, v):
        U = GravityParticleSimulation._particle_interaction_potential_enegry(
            x,self.m,self.G
        )
        return U 
    

    def _get_acceleration(self, x,v,t):
        force = GravityParticleSimulation._get_particle_interaction_force(
            x,self.m,self.G
        )
        acceleration = force/self.m 
        return acceleration
    
    
    def _get_r_square(x,softening=0.1):
        """
        Give the multual distance of particles
        
        Parameters:
            x: a 2d array with shape (n_dim,n_particles)
        Returns:
            r_square: a 2d array with shape (n_particles, n_particles). 
        """
        n_dims = x.shape[0]
        n_particles = x.shape[1]
        
        r_square = np.zeros((n_particles,n_particles))
        for dim in range(n_dims):
            dx = np.subtract.outer(x[dim],x[dim])
            r_square += dx**2
        r_square += softening**2
        return r_square
       

    def _get_particle_interaction_force(x,m,interaction_strength):
        n_dims = x.shape[0]
        n_particles = x.shape[1]
        
        r_square = ParticleSimulation._get_r_square(x)
        inv_r3 = np.zeros((n_particles,n_particles))
        inv_r3[r_square > 0] = r_square[r_square > 0]**(-3.0/2.0)
        
        F = np.zeros((n_dims, n_particles, n_particles))
        for dim in range(n_dims):
            dx = np.subtract.outer(x[dim],x[dim])
            F[dim] = (interaction_strength* dx * inv_r3) * (np.outer(m,m))
            assert np.abs(np.diag(F[dim])).max()==0
        
        max_F = 100
        F[F > max_F] = max_F
        F[F < -max_F] = -max_F
        f = -F.sum(axis=-1)
        return f


        
        
        
        