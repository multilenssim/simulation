from chroma.event import Photons
import nog4_sim as setup
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle as p
import numpy as np
import config_stat
import paths

cfg = 'cfSam1_K200_8_small'
amount = 10000
sim, analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg),useGeant4=False)
det_res = analyzer.det_res
pos = (np.random.rand(amount,3)-0.5)*0.05*det_res.inscribed_radius
drc = config_stat.normalize(np.tile(det_res.lens_centers,amount/len(det_res.lens_centers)).reshape(-1,3),1)             #config_stat.normalize(2*np.random.rand(amount,3)-1,1)
pol = np.cross(drc,np.random.rand(amount,3))
wavelengths = np.repeat(300.0,amount)
gun = Photons(pos,drc,pol,wavelengths)
u_vers = config_stat.normalize(np.cross(np.random.rand(det_res.n_lens_sys,3),det_res.lens_centers),1)
v_vers = config_stat.normalize(np.cross(u_vers,det_res.lens_centers),1)

for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
    b_dir = ev.photons_beg.dir
    _,arr_bin,detected = analyzer.generate_tracks(ev,heat_map=True,detec=True)

b_dir = b_dir[detected]
lns_index = arr_bin/det_res.n_pmts_per_surf
lens_centers = det_res.lens_centers[lns_index]
u_dir = np.einsum('ij,ij->i',b_dir,u_vers[lns_index])
v_dir = np.einsum('ij,ij->i',b_dir,v_vers[lns_index])
FOV = 42*np.pi/180
h_max = det_res.detector_r*np.arcsin(det_res.lns_rad/det_res.detector_r)
c_theta = np.einsum('ij,ij->i',b_dir,config_stat.normalize(lens_centers,1))
c_theta[c_theta>1] = 1
theta = np.arccos(c_theta)
h_points = theta/FOV*h_max
dir_2d = config_stat.normalize(np.einsum('ij,i->ij',u_vers[lns_index],u_dir) + np.einsum('ij,i->ij',v_vers[lns_index],v_dir),1)
perf_points = np.einsum('ij,i->ij',config_stat.normalize(lens_centers,1),np.linalg.norm(lens_centers,axis=1)+det_res.focal_length)+np.einsum('ij,i->ij',dir_2d,h_points)
perf_points_pixel = det_res.pmt_bin_to_position(det_res.find_pmt_bin_array(perf_points))
pmt_pos = det_res.pmt_bin_to_position(arr_bin)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(pmt_pos[:,0],pmt_pos[:,1],pmt_pos[:,2],s=10)
ax.scatter(perf_points_pixel[:,0],perf_points_pixel[:,1],perf_points_pixel[:,2],s=5)
plt.show()
