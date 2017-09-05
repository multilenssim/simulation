from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge,Circle,Arrow
import matplotlib.gridspec as gridspec
from kabamland2 import gaussian_sphere
import generate_simulation as gs
import matplotlib.pyplot as plt
import detectorconfig as dc
import right_amount
import numpy as np

def surf(rad_ring,ring_par,width_ring):
	patches = []	
	for i,e in enumerate(rad_ring):
		theta = np.linspace(0,360,ring_par[2][i]+1)
		for th1,th2 in zip(theta,theta[1:]):
			patches.append(Wedge((0.,0.),e,th1,th2,width=width_ring[i]))
	return PatchCollection(patches)

def tr_center():
	base = conf_par.base
	count = 0
	x,y = [],[]
	while (base>0):
		x = np.append(x,count+np.sqrt(3) + 2*(np.arange(base)))
		y = np.append(y,np.full(base,1+count*np.sqrt(3)))
		base -= 1
		count += 1
	return x,y

def triangle(lns):
	patches = []
	x_c,y_c = tr_center()
	x_cent = np.mean(x_c)
	y_cent = np.mean(y_c)
	for x,y in zip(x_c,y_c):
		patches.append(Circle((x,y),1))
	patches.append(Arrow(x_cent,y_cent,x_c[lns]-x_cent,y_c[lns]-y_cent,width=0.2))
	patches.append(Arrow(x_cent,y_cent,y_c[lns]-y_cent,x_cent-x_c[lns],width=0.2))
	return PatchCollection(patches)

def rot_ax(vrs,arr):
	v = np.cross(vrs,arr)
	cs = np.dot(vrs,arr)
	vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	rtn = np.identity(3)+vx+np.linalg.matrix_power(vx,2)*(1.0/(1.0+cs))
	return rtn

def plot_heat(conf_par,heat,lns,l_rad,):
	max_rad = conf_par.edge_length/(2*(conf_par.base+np.sqrt(3)-1))
	ring_par = right_amount.curved_surface2(conf_par.detector_r,2*max_rad,conf_par.nsteps,conf_par.b_pixel)
	rad_ring = ring_par[0][:,0]/ring_par[0][0,0]
	width_ring = np.absolute(np.diff(rad_ring))
	width_ring = np.append(width_ring,rad_ring[-1])
	fig = plt.figure()
	gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1],height_ratios=[2,1,1])
	ax1 = plt.subplot(gs[:,0],xlim=(-1,1),ylim=(-1,1),xticklabels=[],yticklabels=[],title='Pixelated Heat Map')
	ax2 = plt.subplot(gs[0,1],xlim=(0.5,9),ylim=(0,8.5),xticklabels=[],yticklabels=[], title='Optical System Position')
	ax3 = plt.subplot(gs[1,1],title='Event Topology (side view)')
	ax4 = plt.subplot(gs[2,1],title='Event Topology (top view)')
	p = surf(rad_ring,ring_par,width_ring)
	ax1.add_collection(p)
	p.set_array(heat)
	lns_ix = np.zeros(sys_per_face+2)
	lns_ix[lns] = 1
	lns_ix[-2:] = 0.5
	t = triangle(lns)
	ax2.add_collection(t)
	t.set_array(lns_ix)
	pl3 = locate(dir_to_lens,lens_center[lns],l_rad,lg,1)
	ax3.plot(pl3[3][0],pl3[3][1],linewidth=3)
	ax3.plot(pl3[2][0],pl3[2][1],'o',markersize=15)
	ax3.plot(pl3[0][0],pl3[0][1],linewidth=5)
	ax3.plot(0,0,'x',markersize=15)
	#ax3.plot(pl3[1][0],pl3[1][1],'--')
	ax3.set_xticks([])
	ax3.set_yticks([])
	ax3.set_aspect('equal')
	pl4 = locate(dir_to_lens,lens_center[lns],l_rad,lg,2)
        ax4.plot(pl4[3][0],pl4[3][1],linewidth=3)
	ax4.plot(pl4[2][0],pl4[2][1],'o',markersize=15)
	ax4.plot(pl4[0][0],pl4[0][1],linewidth=5)
	ax4.plot(0,0,'x',markersize=15)
	#ax4.plot(pl4[1][0],pl4[1][1],'--')
	ax4.set_xticks([])
	ax4.set_yticks([])
	ax4.set_aspect('equal')
	fig.colorbar(p, ax=ax1)
	plt.show()

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

def locate(i_rad,top_lens,l_rad,source,proj):
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
	coord = np.matmul(rr_matrix,tria_proj(i_rad,lens_center,l_rad,conf_par.base).T+i_rad)
	if proj == 1:
		twod_lens = np.asarray([np.matmul(rr_matrix,top_lens-l_rad*y_rot), np.matmul(rr_matrix,top_lens+l_rad*y_rot)])
		return [twod_lens[:,0],twod_lens[:,1]], [[0,np.matmul(rr_matrix,i_rad)[0]],[0,np.matmul(rr_matrix,i_rad)[1]]], [r_src[0],r_src[1]], [coord[0],coord[1]]
	if proj == 2:
		twod_lens = np.asarray([np.matmul(rr_matrix,top_lens-l_rad*z_rot), np.matmul(rr_matrix,top_lens+l_rad*z_rot)])
		return [twod_lens[:,0],twod_lens[:,2]], [[0,np.matmul(rr_matrix,i_rad)[0]],[0,np.matmul(rr_matrix,i_rad)[2]]], [r_src[0],r_src[2]], [coord[0],coord[2]]

def tria_proj(i_rad,lens_center,l_rad,base):
	a = (lens_center[0] - i_rad)*(1+2*l_rad/np.linalg.norm(lens_center[0] - i_rad))
	b = (lens_center[base-1] - i_rad)*(1+2*l_rad/np.linalg.norm(lens_center[base-1] - i_rad))
	c = (lens_center[-1] - i_rad)*(1+2*l_rad/np.linalg.norm(lens_center[-1] - i_rad))
	return np.asarray([a,b,c])


if __name__=='__main__':
	cfg = 'cfJiani3_9'
	conf_par = dc.configdict[cfg]
	sys_per_face = (conf_par.base*(conf_par.base+1))/2
	heat = []
	sigma = 0.01
	amount = 1000000
	ix = 4
	sim,analyzer = gs.sim_setup(cfg,'/home/miladmalek/TestData/detresang-'+cfg+'_1DVariance_100million.root')
	pmts_per_surf = analyzer.det_res.n_pmts_per_surf
	lens_center = analyzer.det_res.lens_centers[:sys_per_face]
	dir_to_lens = np.mean(lens_center,axis=0)
	dir_to_lens_n = dir_to_lens/np.linalg.norm(dir_to_lens)
	off_ax = gs.sph_scatter(1,in_shell = 0,out_shell = 1000).T #2000*rand_perp(dir_to_lens_n)
	for lg in ([0,0,0],2000*dir_to_lens_n,-2000*dir_to_lens_n,off_ax,-off_ax):
		sim_event = gaussian_sphere(lg, sigma, amount)			
		for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks,pmt_arr = analyzer.generate_tracks(ev,heat_map=True)
		l_rad = tracks.lens_rad
		#plot_heat(conf_par,color(pmt_arr,ix),ix,l_rad)
		for i in xrange(sys_per_face):
			plot_heat(conf_par,color(pmt_arr,i),i,l_rad)
		break
		
