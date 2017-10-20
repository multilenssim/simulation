from mpl_toolkits.mplot3d import Axes3D
from chroma.generator import vertex
import matplotlib.pyplot as plt
import h5py,time,argparse
import nog4_sim as setup


def gen_ev(sample,cfg,particle,energy,i_r,o_r):
	seed_loc = 'r%i-%i'%(i_r,o_r)
	fname = '/home/jacopodalmasson/Desktop/dev/'+cfg+'/raw_data/'+seed_loc+'_'+str(energy)+particle+'_sim.h5'
	sim,analyzer = setup.sim_setup(cfg,'/home/miladmalek/TestData/detresang-'+cfg+'_1DVariance_100million.root')
	print 'configuration loaded'
	location = setup.sph_scatter(sample,i_r*1000,o_r*1000)
	arr_tr, arr_depo = [],[]
	with h5py.File(fname,'w') as f:
		first = True
		for lg in location:
			gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
			for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
				vert = ev.photons_beg.pos
				tracks = analyzer.generate_tracks(ev,qe=(1./3.))
			if first:
				en_depo = f.create_dataset('en_depo',data=vert,chunks=True)
				coord = f.create_dataset('coord',data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
				uncert = f.create_dataset('sigma',data=tracks.sigmas,chunks=True)
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
			first = False
		f.create_dataset('idx_tr',data=arr_tr)
		f.create_dataset('idx_depo',data=arr_depo)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('particle', help='particle to simulate')
	parser.add_argument('s_d', help='seed location')
	parser.add_argument('cfg', help='detector configuration')	
	args = parser.parse_args()
	sample = 500
	particle = args.particle
	cfg = args.cfg
	s_d = args.s_d
	energy = 2
	start_time = time.time()
	gen_ev(sample,cfg,particle,energy,int(s_d[0]),int(s_d[1]))
	print time.time()-start_time
