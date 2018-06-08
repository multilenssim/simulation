import nog4_sim as setup
from mpl_toolkits.mplot3d import Axes3D
from chroma.event import Photons
import matplotlib.pyplot as plt
import config_stat
import numpy as np
import kabamland2
import paths

def ring_return(arr_bin,ring):
    ring_arr = np.zeros((arr_bin.shape))
    for i,px in enumerate(arr_bin):
        ring_th = px/np.cumsum(ring)
        try:
            ring_arr[i] = np.where(ring_th==0)[-1][0]
        except IndexError:
            ring_arr[i] = 0
    return ring_arr.astype(int)


cfg = 'cfSam1_K200_10_small'
sim, analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg),useGeant4=False)
det_res = analyzer.det_res
pmt_center = det_res.pmt_bin_to_position(np.arange(det_res.n_pmts_per_surf))
l_c = det_res.lens_centers[0]
amount = 2000
l_r = det_res.lens_rad
norm_dir = np.cross(np.random.rand(3),l_c)
u_vers = norm_dir/np.linalg.norm(norm_dir)
v_vers = np.cross(u_vers,l_c/np.linalg.norm(l_c))
u_pmt = np.einsum('ij,j->i',pmt_center,u_vers)
v_pmt = np.einsum('ij,j->i',pmt_center,v_vers)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.scatter(u_pmt,v_pmt,s=0.5,c='black')
sample = 15
off_center = np.einsum('ij,i->ij',np.tile(u_vers,sample).reshape(sample,3),np.linspace(0,np.linalg.norm(det_res.lens_centers[0]),sample))
for alpha in off_center:
    off_axis = np.random.rand(amount,3)*2-1
    drc = np.tile((l_c+alpha)/np.linalg.norm(l_c+alpha),amount).reshape(amount,3)
    pos = np.einsum('ij,i->ij',np.einsum('ij,i->ij',off_axis,1/np.linalg.norm(off_axis,axis=1)),np.linspace(0,det_res.lens_rad,amount))*0.5-alpha
    pol = np.cross(drc,np.random.rand(amount,3))
    wavelengths = np.repeat(300.0,amount)
    gun = Photons(pos,drc,pol,wavelengths)

    for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
        #b_pos = ev.photons_beg.pos
        e_pos = ev.photons_end.pos
        _,_,detected = analyzer.generate_tracks(ev,heat_map=True,detec=True)

    e_pos = e_pos[detected]
    msk = np.einsum('ij,j->i',e_pos,l_c)>0
    e_pos = e_pos[msk]
    u_proj = np.einsum('ij,j->i',e_pos,u_vers)
    v_proj = np.einsum('ij,j->i',e_pos,v_vers)
    ax1.scatter(u_proj,v_proj,s=0.5)
    ax1.set_xlim(-det_res.lns_rad,det_res.lns_rad)
    ax1.set_ylim(-det_res.lns_rad,det_res.lns_rad)
    ax2.scatter(np.arctan(np.linalg.norm(alpha)/np.linalg.norm(l_c))/np.pi,float(u_proj.shape[0])/amount,c='b')
    ax2.set_xlabel('incident angle [$\pi^{-1}$]')
    ax2.set_ylabel('collection efficiency')
plt.show()
