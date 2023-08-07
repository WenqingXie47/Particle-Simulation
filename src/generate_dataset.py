
from simulation.charged_particle_simulation import ChargedParticleSimulation
from simulation.gravity_particle_simulation import GravityParticleSimulation
import time
import numpy as np
import argparse
import os

"""
nbody: python -u generate_dataset.py  --num-train 50000 --sample-freq 500 2>&1 | tee log_generating_100000.log &

nbody_small: python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small 2>&1 | tee log_generating_10000_small.log &

"""

def generate_dataset(sim, n_sims, n_iters, sample_freq):
    loc_all = []
    vel_all = []
    times_all = []
    edges_all = []
    charges_all = []
    masses_all = []
    
    for i in range(n_sims):
        t = time.time()
        sim.randomly_init()
        x_record, v_record, t_record = sim.sample_trajectory(
            n_iters=n_iters,
            sample_freq=sample_freq
        )
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(x_record)
        vel_all.append(v_record)
        times_all.append(t_record)
        edges_all.append(sim.get_edges())
        charges_all.append(sim.get_charges())
        masses_all.append(sim.get_masses())

    charges_all = np.stack(charges_all)
    masses_all = np.stack(masses_all)
    edges_all = np.stack(edges_all)
    times_all = np.stack(times_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)

    return loc_all, vel_all, times_all, charges_all, masses_all, edges_all



parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=5,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=10000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_particles', type=int, default=6,
                    help='Number of particles in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--initial_vel', type=int, default=1,
                    help='consider initial velocity')
parser.add_argument('--sufix', type=str, default="",
                    help='add a sufix to the name')

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16
np.random.seed(args.seed)



sim_name = f"{args.simulation}_{args.n_particles}particles_{args.initial_vel}initvel"
print(sim_name)
sim_dir = os.path.join(*["..","data",sim_name])
if not os.path.isdir(sim_dir):
   os.makedirs(sim_dir)




if __name__ == "__main__":

    # if args.simulation == 'springs':
    #     sim = SpringSim(n_particles=args.n_particles)
    if args.simulation == 'charged':
        sim = ChargedParticleSimulation(n_particles=args.n_particles)
    elif args.simulation == 'gravity':
        sim = GravityParticleSimulation(n_particles=args.n_particles)
    else:
        raise ValueError(f'Simulation {args.simulation} not implemented')

    
    
    # Generate train data
    print(f"Generating {args.num_train} training simulations")
    loc_train, vel_train, time_train, charges_train, masses_train, edges_train = generate_dataset(
        sim,
        args.num_train,
        args.length,
        args.sample_freq
    )
    np.save(os.path.join(sim_dir,'loc_train.npy'), loc_train)
    np.save(os.path.join(sim_dir,'vel_train.npy'), vel_train)
    np.save(os.path.join(sim_dir,'time_train.npy'), time_train)
    np.save(os.path.join(sim_dir,'charges_train.npy'), charges_train)
    np.save(os.path.join(sim_dir,'masses_train.npy'), masses_train)
    np.save(os.path.join(sim_dir,'edges_train.npy'), edges_train)
    
    
    # Generate validation data
    print(f"Generating {args.num_valid} validation simulations")
    loc_val, vel_val, time_val, charges_val, masses_val, edges_val = generate_dataset(
        sim,
        args.num_valid,
        args.length,
        args.sample_freq
    )
    np.save(os.path.join(sim_dir,'loc_valid.npy'), loc_val)
    np.save(os.path.join(sim_dir,'vel_valid.npy'), vel_val)
    np.save(os.path.join(sim_dir,'time_valid.npy'), time_val)
    np.save(os.path.join(sim_dir,'charges_valid.npy'), charges_val)
    np.save(os.path.join(sim_dir,'masses_valid.npy'), masses_val)
    np.save(os.path.join(sim_dir,'edges_valid.npy'), edges_val)
    

    # Generate test data
    print(f"Generating {args.num_test} test simulations")
    loc_test, vel_test, time_test, charges_test, masses_test, edges_test = generate_dataset(
        sim,
        args.num_test,
        args.length,
        args.sample_freq
    )
    np.save(os.path.join(sim_dir,'loc_test.npy'), loc_test)
    np.save(os.path.join(sim_dir,'vel_test.npy'), vel_test)
    np.save(os.path.join(sim_dir,'time_test.npy'), time_test)
    np.save(os.path.join(sim_dir,'charges_test.npy'), charges_test)
    np.save(os.path.join(sim_dir,'masses_test.npy'), masses_test)
    np.save(os.path.join(sim_dir,'edges_test.npy'), edges_test)