from chroma.generator import vertex

import h5py,time,argparse
import os
import sys

import pycuda.driver as cuda
import paths
import nog4_sim

from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import ThreadPool
import multiprocessing          # Just for CPU count

def cuda_stat():
    cuda.init()
    ndevices = cuda.Device.count()
    print('CUDA/GPU device count: ' + str(ndevices))
    # Can we assume that they are in linear order??
    return ndevices

def gen_ev(sample,cfg,particle,energy,i_r,o_r, cuda_device=None):
	seed_loc = 'r%i-%i'%(i_r,o_r)
        data_file_dir = paths.get_data_file_path(cfg)
        if not os.path.exists(data_file_dir):
	        os.makedirs(data_file_dir)
        fname = data_file_dir+seed_loc+'_'+str(energy)+particle+'_'+'sim.h5'
	sim,analyzer = nog4_sim.sim_setup(cfg, paths.get_calibration_file_name(cfg), useGeant4=True, cuda_device=cuda_device)
	print('Configuration loaded: ' + cfg)
        print('Particle: ' + particle)
        print('Energy: ' + str(energy))
        print('Distance range: ' + str(i_r) + ' ' + str(o_r))
	location = nog4_sim.sph_scatter(sample,i_r*1000,o_r*1000)
	arr_tr, arr_depo = [],[]
	with h5py.File(fname,'w') as f:
	        first = True
                print('Run count: ' + str(len(location)))
                for lg in location:
                        start = time.time()
			gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
			for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
				vert = ev.photons_beg.pos
				tracks = analyzer.generate_tracks(ev,qe=(1./3.))
			if first:
				en_depo = f.create_dataset('en_depo',maxshape=(None,3),data=vert,chunks=True)
				coord = f.create_dataset('coord',maxshape=(2,None,3),data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
				uncert = f.create_dataset('sigma',maxshape=(None,),data=tracks.sigmas,chunks=True)
				f.create_dataset('r_lens',data=tracks.lens_rad)
			else:
				en_depo.resize(en_depo.shape[0]+vert.shape[0], axis=0)
				en_depo[-vert.shape[0]:,:] = vert
				coord.resize(coord.shape[1]+tracks.means.shape[1], axis=1)
				coord[:,-tracks.means.shape[1]:,:] = [tracks.hit_pos.T, tracks.means.T]
				uncert.resize(uncert.shape[0]+tracks.sigmas.shape[0], axis=0)
				uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas
			arr_depo.append(en_depo.shape[0])
			arr_tr.append(uncert.shape[0])
                        print ('Time: ' + str(time.time() - start) + '\tPhotons detected: ' + str(tracks.sigmas.shape[0])) 
			first = False
		f.create_dataset('idx_tr',data=arr_tr)
		f.create_dataset('idx_depo',data=arr_depo)

def run_simulation(cfg, particle, dist_range, energy):
        run_simulation_with_device(cfg, particle, dist_range, energy, None)

def run_simulation_with_device(cfg, particle, dist_range, energy, cuda_device):
        #sys.stdout = open(str(os.getpid()) + ".out", "a", buffering=0)
        #sys.stderr = open(str(os.getpid()) + "_error.out", "a", buffering=0)
        #print('Initializinging CUDA in subprocess')
        try:
                #cuda.init()   # This may be necessary for any sub-process because it is done during some import otherwise....
                pass
        except Exception as e:
                print('Exception raised initializing CUDA in subproces: ' + str(e))
                exit(-1)
        sample = 500
        start_time = time.time()
        print('CUDA initialized')
        gen_ev(sample, cfg, particle, energy, int(dist_range[0]), int(dist_range[1]), cuda_device=None)
        print('Simulation time: ' + str(time.time() - start_time))
        sys.stdout.flush()
        sys.stderr.flush()

if __name__=='__main__':
        parser = argparse.ArgumentParser()
        #parser.add_argument('particle', help='particle to simulate')
        #parser.add_argument('s_d', help='seed location')
        parser.add_argument('cfg', help='detector configuration')
        args = parser.parse_args()

        #cuda_devs = cuda_stat()
        # Just hardwire hack in the device number for now
        next_device = 0

        pool = Pool(multiprocessing.cpu_count())
        energy = 20.
        for particle in ['neutron']: # ['e-']:  # ,'gamma']:
                for dist_range in ['01']:  #,'34']:
                        #pool.apply_async(run_simulation_with_device, (args.cfg, particle, dist_range, next_device))
                        run_simulation_with_device(args.cfg, particle, dist_range, energy, next_device)
                        next_device += 1
        time.sleep(4)  # Trying to catch errors
        pool.close()
        pool.join()
