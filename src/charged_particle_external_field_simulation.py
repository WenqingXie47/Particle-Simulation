


from .charged_particle_simulation import ChargedParticleSimulation
import numpy as np

class ExternalFieldChargedParticleSimulation(ChargedParticleSimulation):
    
    def __init__(self, dim=3, box_size=5, n_particles=5, interaciton_strength=1):
        super().__init__(dim, box_size, n_particles)
        self.interaction_strength=interaciton_strength
        
        
    def init_field(self, amplitude, frequency, phase):
        self.field_amplitude = amplitude
        self.field_frequency = frequency
        self.field_phase = phase
    
    
    def randomly_init_field(self, amp_std=1, log_f_std=1, phase_std=1):
        self.field_amplitude = np.random.randn(self.dim) * amp_std
        self.field_frequency = np.exp(np.random.randn()*log_f_std)
        self.field_phase = np.random.randn()* phase_std
        
    def _get_field(self, t):
        field =  self.field_amplitude*np.cos(self.field_frequency*t+self.field_phase)
        return field
    
    def _field_potential_energy(x,e,field):
        U = 0 
        n_particles = x.shape[1]
        for i in range(n_particles):
            U += -np.dot(x[:,i],field) * e[i]
        return U
            
    def _potential_energy(self, x, v, t):
        
        U = ChargedParticleSimulation._particle_interaction_potential_enegry(
            x,self.e,self.interaction_strength
        )
        
        field = self._get_field(t)
        U+= ExternalFieldChargedParticleSimulation._field_potential_energy(x,self.e,field)
        return U 
    
    def _energy(self, x, v, t):
        K = self._kinetic_energy(v)
        U = self._potential_energy(x,v,t)
        return U + K
    
    
    def _get_acceleration(self, x,v,t):
        force = ChargedParticleSimulation._get_particle_interaction_force(
            x,self.e,self.interaction_strength
        )
        field = self._get_field(t)
        force += ExternalFieldChargedParticleSimulation._get_field_force(self.e,field)
        acceleration = force/self.m 
        return acceleration
    
    
    def _get_field_force(e,field):
        return np.outer(field,e)
        