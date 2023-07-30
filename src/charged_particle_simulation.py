
from .particle_simulation import ParticleSimulation
import numpy as np

class ChargedParticleSimulation(ParticleSimulation):
    
    def __init__(self, dim=3, box_size=5, n_particles=5, interaciton_strength=1):
        super().__init__(dim, box_size, n_particles)
        self.interaction_strength=interaciton_strength
        
        
    def randomly_init_particles(self,x_std=0.5,v_std=0.5,m_std=0.5,e_std=0.5):
        super().init_particles(x_std,v_std)
        self.m = np.random.randn(self.n_particles) * m_std
        self.e = np.random.randn(self.n_particles) * e_std
    
    def init_particles(self,x,v,m,e):
        self.x = x
        self.v = v
        self.m = m
        self.e = e    
    
    def _particle_interaction_potential_enegry(x,e,interaction_strength):
        U=0
        for i in range(x.shape[1]):
            for j in range(i):
                r = x[:, i] -x[:, j]
                r_norm = np.sqrt((r ** 2).sum())
                U += interaction_strength * e[i] * e[j] / r_norm
        return U
    
    def _kinetic_energy(self, v):
        K = ((1/2) * self.m * (v ** 2)).sum()
        return K
    
    def _potential_energy(self, x, v):
        U = ChargedParticleSimulation._particle_interaction_potential_enegry(
            x,self.e,self.interaction_strength
        )
        return U 
    
    def _energy(self, x, v):
        K = self._kinetic_energy(v)
        U = self._potential_energy(x,v)
        return U + K
    
    def _get_acceleration(self, x,v,t):
        force = ChargedParticleSimulation._get_particle_interaction_force(
            x,self.e,self.interaction_strength
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
       

    def _get_particle_interaction_force(x,e,interaction_strength):
        n_dims = x.shape[0]
        n_particles = x.shape[1]
        
        r_square = ChargedParticleSimulation._get_r_square(x)
        inv_r3 = np.zeros((n_particles,n_particles))
        inv_r3[r_square > 0] = r_square[r_square > 0]**(-3.0/2.0)
        
        F = np.zeros((n_dims, n_particles, n_particles))
        for dim in range(n_dims):
            dx = np.subtract.outer(x[dim],x[dim])
            F[dim] = (interaction_strength* dx * inv_r3) * (np.outer(e,e))
            assert np.abs(np.diag(F[dim])).max()==0
        
        max_F = 100
        F[F > max_F] = max_F
        F[F < -max_F] = -max_F
        f = F.sum(axis=-1)
        return f


        
        
        
        