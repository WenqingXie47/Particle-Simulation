import numpy as np


class ParticleSimulation:

    def __init__(self, dim=3,box_size=5, n_particles=5):
        self.dim=dim
        self.box_size = box_size
        self.n_particles = n_particles
        
    
    def init_particles(self,x,v):
        self.x = x
        self.v = v
        
    def randomly_init_particles(self,x_std=0.5,v_std=0.5):
        self.x = np.random.randn(self.dim, self.n_particles) * x_std
        self.v = np.random.randn(self.dim, self.n_particles) * v_std

    
    
    def sample_trajectory(self, dT=0.001, n_iters=10000, sample_freq=10, noise_std=0):

        assert (n_iters % sample_freq == 0)

        # Initialize location and velocity
        n_record = int(n_iters / sample_freq)
        x_record = np.zeros((n_record , self.dim, self.n_particles))
        v_record = np.zeros((n_record , self.dim, self.n_particles))
        t_record = np.zeros((n_record))

        x,v = self.x, self.v
        counter = 0
        t=0
        for i in range(0, n_iters):
            x,v = ParticleSimulation._clamp(x,v,self.box_size)
            if i % sample_freq == 0:
                x_record[counter, :, :], v_record[counter, :, :], t_record[counter] = x, v, t
                counter += 1
            x,v,t = ParticleSimulation._leap_frog(x,v,t,dT,self._get_acceleration)
  
        x_noisy, v_noisy = ParticleSimulation._add_noise_to_observations(x_record,v_record,noise_std)
        return x_noisy, v_noisy, t_record


    def _get_acceleration(self,x,v,t):
        return np.zeros(x.shape)
    
    def _kinetic_energy(self, v):
        K = 0.5 * (v ** 2).sum()
        return K
    
    def _potential_energy(self, x, v):
        return 0
    
    def _energy(self, x, v):
        K = self._kinetic_energy(v)
        U = self._potential_energy(x,v)
        return U + K
    
    
    def _leap_frog(x,v,t,dT,get_acceleration):
        a = get_acceleration(x,v,t)
        v_mid = v + a * dT/2.0
        x_next = x + v_mid * dT
        a_next = get_acceleration(x,v,t)
        v_next = v_mid + a_next * dT/2.0
        t_next = t+dT
        return x_next, v_next, t_next
    
    
    def _clamp(x, v, box_size):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(x < box_size * 3))
        assert (np.all(x > -box_size * 3))

        over = x > box_size
        x[over] = 2 * box_size - x[over]
        assert (np.all(x <= box_size))

        # assert(np.all(vel[over]>0))
        v[over] = -np.abs(v[over])

        under = x < -box_size
        x[under] = -2 * box_size - x[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(x >= -box_size))
        v[under] = np.abs(v[under])
        return x, v


    def _add_noise_to_observations(x_record, v_record, noise_std=0):
        x_noisy = x_record + np.random.randn(*x_record.shape) * noise_std
        v_noisy = v_record + np.random.randn(*v_record.shape) * noise_std
        return x_noisy, v_noisy
    
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
    


