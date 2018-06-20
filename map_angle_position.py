from chroma.generator import vertex
from chroma.event import Photons
from Tracks import Tracks
import utilities as setup
import detectorconfig
import pickle as p
import numpy as np
import config_stat
import inspect
import paths
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class perf_resolution(object):

    def __init__(self, cfg, gun, fov, qe=1.0/3.0):
        self._cfg = cfg
        self.u_vers = None
        self.v_vers = None
        self.b_dir = None
        self._fov = fov

        if type(gun).__name__ == 'generator': g4 = True
        elif type(gun).__name__ == 'Photons': g4 = False

        sim, analyzer = setup.sim_setup(self._cfg,paths.get_calibration_file_name(self._cfg.config_name),useGeant4=g4)
        self._det_res = analyzer.det_res

        for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
            _,self._arr_bin,detected = analyzer.generate_tracks(ev,qe,heat_map=True,detec=True)

        self.u_vers = config_stat.normalize(np.cross(np.random.rand(self._det_res.n_lens_sys,3),self._det_res.lens_centers),1)
        self.v_vers = config_stat.normalize(np.cross(self.u_vers,self._det_res.lens_centers),1)
        self.b_dir = analyzer.photons_beg_dir[detected]
        self._lns_index = self._arr_bin/self._det_res.n_pmts_per_surf

    def get_points(self):
        lens_centers = -self._cfg.vtx[self._lns_index]
        u_dir = np.einsum('ij,ij->i',self.b_dir,self.u_vers[self._lns_index])
        v_dir = np.einsum('ij,ij->i',self.b_dir,self.v_vers[self._lns_index])
        FOV = self._fov*np.pi/180
        h_max = self._det_res.detector_r*np.arcsin(self._det_res.lns_rad/self._det_res.detector_r)
        c_theta = np.einsum('ij,ij->i',self.b_dir,config_stat.normalize(lens_centers,1))
        c_theta[c_theta>1] = 1
        theta = np.arccos(c_theta)
        h_points = theta/FOV*h_max
        dir_2d = config_stat.normalize(np.einsum('ij,i->ij',self.u_vers[self._lns_index],u_dir) + np.einsum('ij,i->ij',self.v_vers[self._lns_index],v_dir),1)
        perf_points = np.einsum('ij,i->ij',config_stat.normalize(lens_centers,1),np.linalg.norm(lens_centers,axis=1)+self._det_res.focal_length) + np.einsum('ij,i->ij',dir_2d,h_points)
        large_angle = theta>FOV
        perf_points[large_angle] = self._det_res.pmt_bin_to_position(self._arr_bin[large_angle])
        curved_points = np.einsum('ij,i->ij',config_stat.normalize(perf_points,1),self.__curved_projection(lens_centers,perf_points))
        return curved_points

    def __curved_projection(self, lens_centers,perf_points):
        cos_gamma = np.einsum('ij,ij->i',config_stat.normalize(lens_centers,1),config_stat.normalize(perf_points,1))
        a = np.linalg.norm(lens_centers,axis=1)+self._det_res.focal_length - self._det_res.detector_r
        c = self._det_res.detector_r
        return a*cos_gamma + np.sqrt(c*c - a*a*(1-np.square(cos_gamma)))


    def tracks(self):
        return Tracks(self._det_res.lens_centers[self._lns_index].T, self.b_dir.T, np.zeros(len(self.b_dir)))


if __name__ == '__main__':

    cfg = detectorconfig.get_detector_config('cfSam1_l200_p107600_b4_e10')
    amount = 16000
    pos = (2*np.random.rand(amount,3)-1)
    drc = config_stat.normalize(np.tile(-cfg.vtx,amount/len(cfg.vtx)).reshape(-1,3),1)
    pol = np.cross(drc,np.random.rand(amount,3))
    wavelengths = np.repeat(300.0,amount)
    gun = Photons(pos,drc,pol,wavelengths)
    energy = 2
    #gun = vertex.particle_gun(['e-'], vertex.constant([0,0,0]), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
    a = perf_resolution(cfg,gun,42)
    tr = a.tracks()
    print tr.hit_pos.shape,tr.means.shape,tr.sigmas.shape
    #curved_points = a.get_points()

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.scatter(curved_points[:,0],curved_points[:,1],curved_points[:,2],s=5)
    #plt.show()
