import numpy as np
import argparse

from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge,Circle,Arrow
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from chroma.generator import vertex
from chroma.event import Photons
from chroma import sample

from kabamland2 import gaussian_sphere
import detectorconfig
import right_amount
import paths
import utilities

import os
import numpy as np
import argparse

def surf(rad_ring,ring_par,width_ring):
	patches = []	
	for i,e in enumerate(rad_ring):
		theta = np.linspace(0,360,ring_par[2][i]+1)
		for th1,th2 in zip(theta,theta[1:]):
			patches.append(Wedge((0.,0.),e,th1,th2,width=width_ring[i]))
	return PatchCollection(patches)

def tr_center(l_rad, config):
	base = config.lens_count
	count = 0
	x,y = [],[]
	while (base>0):
		x = np.append(x,l_rad*(count+np.sqrt(3) + 2*(np.arange(base))))
		y = np.append(y,l_rad*(np.full(base,1+count*np.sqrt(3))))
		base -= 1
		count += 1
	return x,y

def triangle(lns,l_rad,config):
	patches = []
	x_c,y_c = tr_center(l_rad,config)
	x_cent = np.mean(x_c)
	y_cent = np.mean(y_c)
	for x,y in zip(x_c,y_c):
		patches.append(Circle((x,y),l_rad))
	patches.append(Arrow(x_cent,y_cent,x_c[lns]-x_cent,y_c[lns]-y_cent,width=0.2))
	patches.append(Arrow(x_cent,y_cent,y_c[lns]-y_cent,x_cent-x_c[lns],width=0.2))
	return PatchCollection(patches)

def rot_ax(vrs,arr):
	v = np.cross(vrs,arr)
	cs = np.dot(vrs,arr)
	vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	rtn = np.identity(3)+vx+np.linalg.matrix_power(vx,2)*(1.0/(1.0+cs))
	return rtn

def plot_heat(conf_par,heat,lns,l_rad, particle_name):
	#max_rad = conf_par.edge_length/(2*(conf_par.base+np.sqrt(3)-1))   # May be an old line?  Was in my branch
	max_rad = conf_par.half_EPD/conf_par.EPD_ratio
	ring_par = right_amount.curved_surface2(conf_par.detector_r,2*max_rad,conf_par.ring_count,conf_par.base_pixels)
	rad_ring = ring_par[0][:,0]/ring_par[0][0,0]
	width_ring = np.absolute(np.diff(rad_ring))
	width_ring = np.append(width_ring,rad_ring[-1])
	fig = plt.figure()
	fig.suptitle(conf_par.config_name+' Design: Event Displayer',fontsize=20)
	gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1],height_ratios=[2,1,1])
	ax1 = plt.subplot(gs[:,0],xlim=(-1,1),ylim=(-1,1),xticklabels=[],yticklabels=[],title='Pixelated Heat Map, Photon Detected: '+str(sum(heat)))
	ax2 = plt.subplot(gs[0,1],xlim=(0,10000),ylim=(0,5000*np.sqrt(3)), title='Optical System Position')
	ax3 = plt.subplot(gs[1,1],title='Event Topology (side view)')
	ax4 = plt.subplot(gs[2,1],title='Event Topology (top view)')
	p = surf(rad_ring,ring_par,width_ring)
	ax1.add_collection(p)
	p.set_array(heat)
	lns_ix = np.zeros(200+2)  # sys_per_face+2)
	lns_ix[lns] = 1
	lns_ix[-2:] = 0.5
	t = triangle(lns,l_rad,conf_par)
	ax2.add_collection(t)
	t.set_array(lns_ix)
	ax2.set_aspect('equal')
	pl3 = locate(dir_to_lens,lens_center[lns],l_rad,lg,1,conf_par)
	ax3.plot(pl3[3][0],pl3[3][1],linewidth=3)
	ax3.plot(pl3[2][0],pl3[2][1],'o',markersize=15)
	ax3.plot(pl3[0][0],pl3[0][1],linewidth=5)
	ax3.plot(0,0,'x',markersize=15)
	ax3.set_xticks([])
	ax3.set_yticks([])
	ax3.set_aspect('equal')
	pl4 = locate(dir_to_lens,lens_center[lns],l_rad,lg,2, conf_par)
	ax4.plot(pl4[3][0],pl4[3][1],linewidth=3)
	ax4.plot(pl4[2][0],pl4[2][1],'o',markersize=15)
	ax4.plot(pl4[0][0],pl4[0][1],linewidth=5)
	ax4.plot(0,0,'x',markersize=15)
	ax4.set_xticks([])
	ax4.set_yticks([])
	ax4.set_aspect('equal')
	fig.colorbar(p, ax=ax1)
	fig.set_size_inches(10,6)

	map_path = 'hmaps/'
	if not os.path.exists(map_path):
			os.makedirs(map_path)

	filename = map_path+'heat'+str(lns)+'-'+conf_par.config_name+'-'+particle_name
	#fig.savefig(filename+'.png')
	fig.savefig(filename+'.pdf')

	'''
        F = pylab.gcf()
	ds = F.get_size_inches()
	F.set_size_inches((ds[0]*2.45,ds[1]*1.83))
	F.savefig('heat'+str(lns)+str(lg)[:2]+'.png')
	#plt.show()
        '''

def color(pmt_arr,opt_sys):
	mask = pmt_arr/pmts_per_surf==opt_sys
	lens_idx = pmt_arr[mask]/pmts_per_surf
	pmt_arr = pmt_arr[mask]
	select_pmt = pmt_arr-lens_idx*pmts_per_surf
	heat,bn = np.histogram(select_pmt,bins=np.linspace(0.5,pmts_per_surf+0.5,pmts_per_surf+1))
	return heat

def rand_perp(arr):
	vrs = [0,0,1]
	v = np.cross(vrs,arr)
	cs = np.dot(vrs,arr)
	vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	rtn = np.identity(3)+vx+np.linalg.matrix_power(vx,2)*(1.0/(1.0+cs))
	rd_gen = np.random.rand()
	pl_arr = np.array([rd_gen,np.sqrt(1.0-np.square(rd_gen)),0])
	return np.matmul(rtn,pl_arr)

def locate(i_rad,top_lens,l_rad,source,proj,config):
	x_rot = i_rad/np.linalg.norm(i_rad)
	if np.array_equal(i_rad,top_lens):
		y_rot = rand_perp(x_rot)
	else:
		y_rot = (top_lens-i_rad)/np.linalg.norm(top_lens-i_rad)
	z_rot = np.cross(x_rot,y_rot)
	r_matrix = rot_ax(z_rot,[0,0,1])
	pl_rot = rot_ax(np.matmul(r_matrix,x_rot),[1,0,0])
	rr_matrix = np.matmul(pl_rot,r_matrix)
	r_src = np.matmul(rr_matrix,source)
	coord = np.matmul(rr_matrix,tria_proj(i_rad,lens_center,l_rad,config.lens_count).T+i_rad)
	if proj == 1:
		twod_lens = np.asarray([np.matmul(rr_matrix,top_lens-l_rad*y_rot), np.matmul(rr_matrix,top_lens+l_rad*y_rot)])
		return [twod_lens[:,0],twod_lens[:,1]], [[0,np.matmul(rr_matrix,i_rad)[0]],[0,np.matmul(rr_matrix,i_rad)[1]]], [r_src[0],r_src[1]], [coord[0],coord[1]]
	if proj == 2:
		twod_lens = np.asarray([np.matmul(rr_matrix,top_lens-l_rad*z_rot), np.matmul(rr_matrix,top_lens+l_rad*z_rot)])
		return [twod_lens[:,0],twod_lens[:,2]], [[0,np.matmul(rr_matrix,i_rad)[0]],[0,np.matmul(rr_matrix,i_rad)[2]]], [r_src[0],r_src[2]], [coord[0],coord[2]]

def tria_proj(i_rad,lens_center,l_rad,base):
	a = (lens_center[0] - i_rad)*(1+l_rad/np.linalg.norm(lens_center[0] - i_rad))
	b = (lens_center[base-1] - i_rad)*(1+l_rad/np.linalg.norm(lens_center[base-1] - i_rad))
	c = (lens_center[-1] - i_rad)*(1+l_rad/np.linalg.norm(lens_center[-1] - i_rad))
	return np.asarray([a,b,c])


def line_of_photons(n):
	points = np.empty((n, 3))
	np.linspace(-2, 2, n)
	points[:, 0] = np.linspace(-2, 2, n)
	points[:, 1] = 0.
	points[:, 2] = 0.
	pos = points
	dir = np.empty([n, 3])
	dir[:, 0] = 0.		# there's got to be a better way to do this....
	dir[:, 1] = 1.
	dir[:, 2] = 0.
	pol = np.cross(dir, sample.uniform_sphere(n))  # ??
	# 300 nm is roughly the pseudocumene scintillation wavelength
	wavelengths = np.repeat(300.0, n)
	return Photons(pos, dir, pol, wavelengths)


def gaussian_sphere(pos, sigma, n):
	points = np.empty((n, 3))
	points[:, 0] = np.random.normal(0.0, sigma, n) + pos[0]
	points[:, 1] = np.random.normal(0.0, sigma, n) + pos[1]
	points[:, 2] = np.random.normal(0.0, sigma, n) + pos[2]
	pos = points
	dir = sample.uniform_sphere(n)
	pol = np.cross(dir, sample.uniform_sphere(n))
	# 300 nm is roughly the pseudocumene scintillation wavelength
	wavelengths = np.repeat(300.0, n)
	return Photons(pos, dir, pol, wavelengths)

def print_photons_meta_data(photons):
	print("Gun: " + str(np.shape(gun.pos)) + " " + str(np.shape(gun.dir)) + " " + str(np.shape(gun.pol)) + " " + str(np.shape(gun.wavelengths)))
	print("Gun types: " + str(gun.pos.dtype) + " " + str(gun.dir.dtype) + " " + str(gun.pol.dtype) + " " + str(gun.wavelengths.dtype))

if __name__=='__main__':
	g4 = True
	sigma = 0.01
	amount = 1000 # 1000000
	energy = 2

	gun = gaussian_sphere((0,0,0), sigma, amount)
	gun2 = line_of_photons(amount)
	print_photons_meta_data(gun)
	print_photons_meta_data(gun2)
	print("Photon gun count: " + str(len(gun)))

	parser = argparse.ArgumentParser()
	parser.add_argument('config_name', help='provide configuration')
	args = parser.parse_args()
	config_name = args.config_name
	config = detectorconfig.get_detector_config(config_name)

    # TODO: No such things as systems per face any more
	lens_count = config.lens_count

	sel_len = range(lens_count)  # np.random.choice(sys_per_face,3,replace=False)
	heat = []
	#ix = 4
	sim,analyzer = utilities.sim_setup(config,paths.get_calibration_file_name(config_name), useGeant4=True)  # Do we really want to require turning on Geant4 explicitly?
	pmts_per_surf = analyzer.det_res.n_pmts_per_surf
	lens_center = analyzer.det_res.lens_centers[:lens_count]
	dir_to_lens = np.mean(lens_center,axis=0)
	dir_to_lens_n = dir_to_lens/np.linalg.norm(dir_to_lens)
	off_ax = utilities.sph_scatter(1,in_shell = 0,out_shell = 1000).T #2000*rand_perp(dir_to_lens_n)
	if np.shape(off_ax)[1] == 1:
		off_ax = np.reshape(off_ax,(1,3))
	else:
		pass
	print("Location count: " + str(len(off_ax)))
	for lg in off_ax:
		print("Location: " + str(lg))
		if not g4:
			#gun = gaussian_sphere(lg, sigma, amount)
			gun = line_of_photons(amount)
			print("Photon gun count: " + str(len(gun)))
		else:
			gun = vertex.particle_gun(['e-','gamma'], vertex.constant(lg), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
                if gun is None:    # Sanity check
                        print('Gone is "none"')
		for ev in sim.simulate(gun, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks,pmt_arr = analyzer.generate_tracks(ev,qe=(1./3.),heat_map=True)
			if config.lens_system_name == 'Jiani3':
				l_rad = tracks.lens_rad/0.75835272409
			elif config.lens_system_name == 'Sam1':
				l_rad = tracks.lens_rad
			
			#plot_heat(conf_par,color(pmt_arr,ix),ix,l_rad)
			for i in sel_len:
				plot_heat(config,color(pmt_arr,i),i,l_rad,ev.primary_vertex.particle_name)
		
