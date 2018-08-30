from chroma.generator import vertex
from chroma.event import Photons
from Tracks import Tracks
import utilities as setup
import kabamland2 as kb
import DetectorResponse
import detectorconfig
import argparse,time
import numpy as np
import config_stat
import h5py as h5
import paths



class perf_resolution(object):


    def __init__(self, sim, analyzer, gun, fov, qe=1.0/3.0):
        self._det_res = analyzer.det_res
        self.u_vers = None
        self.v_vers = None
        self.b_dir = None
        self._fov = fov
        

        for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
            _,self._arr_bin,detected = analyzer.generate_tracks(ev,qe,heat_map=True,detec=True)
            self.vert = ev.photons_beg.pos
            self.detected_flag = (ev.photons_end.flags & (0x1<<2)).astype(bool)

        if qe is None: self.b_dir = analyzer.photons_beg_dir
        else: self.b_dir = analyzer.photons_beg_dir[detected]

        self._lns_index = self._arr_bin/self._det_res.n_pmts_per_surf


    def get_points(self):
        self.u_vers = config_stat.normalize(np.cross(np.random.rand(self._det_res.n_lens_sys,3),self._det_res.lens_centers),1)
        self.v_vers = config_stat.normalize(np.cross(self.u_vers,self._det_res.lens_centers),1)
        lens_centers = -self._det_res.config.vtx[self._lns_index]
        u_dir = np.einsum('ij,ij->i',self.b_dir,self.u_vers[self._lns_index])
        v_dir = np.einsum('ij,ij->i',self.b_dir,self.v_vers[self._lns_index])
        FOV = np.radians(self._fov)
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


    def infite_pixel_tracks(self):
        return Tracks(self._det_res.lens_centers[self._lns_index].T, self.b_dir.T, np.zeros(len(self.b_dir)),lens_rad=self._det_res.lens_rad)


    def perf_lens_tracks(self):
        points = self.get_points()
        pmt = self._det_res.find_pmt_bin_array(points)
        lns_index = pmt/self._det_res.n_pmts_per_surf
        return Tracks(self._det_res.lens_centers[lns_index].T, self.read_perf_lens_calibration()[pmt], self._det_res.sigmas[pmt], lens_rad=self._det_res.lens_rad)


    def read_perf_lens_calibration(self):
        with h5.File('%sperf_cal_%s.h5'%(paths.detector_calibration_path,self._det_res.configname),'r') as f:
            return f['means'][:]


def dataset_calibrate(cfg):
    sim, analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg.config_name),useGeant4=False)
    n_loop = 1000
    n_ph = 100000

    with h5.File('%sno_perf_sim_%s.h5'%(paths.detector_calibration_path,cfg.config_name),'w') as f:
        photons_start = f.create_dataset('photons_start', shape=(n_loop*n_ph,3), dtype=np.float32, chunks=True)
        hit_pixel = f.create_dataset('hit_pixel', shape=(n_loop*n_ph,), dtype=int, chunks=True)

        for i in xrange(n_loop):
            if i%50 == 0: print '%i loop'%i
            
            loop_hit = np.full(n_ph,-1)
            gun = kb.uniform_photons(cfg.detector_radius,n_ph)

            #REMOVE THIS PART FOR PERFECT LENS

            for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
                _,arr_bin = analyzer.generate_tracks(ev,qe=None,heat_map=True)
                vert = ev.photons_beg.pos
                detected_flag = (ev.photons_end.flags & (0x1<<2)).astype(bool)

            loop_hit[detected_flag] = arr_bin
            photons_start[i*n_ph:(i+1)*n_ph] = vert

            #UNCOMMENT HERE
             
            #pr = perf_resolution(sim, analyzer, gun, 42.0, qe=None)
            #loop_hit[pr.detected_flag] = analyzer.det_res.find_pmt_bin_array(pr.get_points())
            #photons_start[i*n_ph:(i+1)*n_ph] = pr.vert
            hit_pixel[i*n_ph:(i+1)*n_ph] = loop_hit


def write_perf_lens_calibration(cfg):
    det_res = DetectorResponse.DetectorResponse(cfg)
    sim_name = '%sh5fied_sim.h5'%paths.detector_calibration_path
    #sim_name = '%sno_perf_sim_%s.h5'%(paths.detector_calibration_path,cfg.config_name)
    bin_locations = det_res.pmt_bin_to_position(np.arange(cfg.tot_pixels))
    calibrated_direction = np.zeros((cfg.tot_pixels,3))
    s_time = time.time()

    with h5.File(sim_name,'r') as f:
        hit_pixel = f['hit_pixel'][:]
        photons_start = f['photons_start'][:]

    photons_start = photons_start[hit_pixel!=-1]
    hit_pixel = hit_pixel[hit_pixel!=-1]

    for i in xrange(cfg.tot_pixels):
        if i%1000 == 0: print '%i in %.2f'%(i,time.time()-s_time)

        ix_array = np.where(hit_pixel==i)[0]
        dir_to_lens = photons_start[ix_array] - det_res.lens_centers[i/det_res.n_pmts_per_surf]
        calibrated_direction[i] = np.mean(config_stat.normalize(dir_to_lens,1),axis=0)

    norm_calibrated_direction = config_stat.normalize(calibrated_direction,1)

    with h5.File('%sno_perf_cal_h5fied_%s.h5'%(paths.detector_calibration_path,cfg.config_name),'w') as f:
        f.create_dataset('means', data=norm_calibrated_direction.T)


def test_perf_calibration():
    cfg = detectorconfig.get_detector_config('cfSam1_l200_p107600_b4_e10')
    g4 = False
    amount = 1000
    energy = 2

    if g4:
        gun = vertex.particle_gun(['e-'], vertex.constant([0,0,0]), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))

    else:
        pos = np.random.rand(amount,3)
        drc = config_stat.normalize(2*np.random.rand(amount,3)-1,1)
        pol = np.cross(drc,np.random.rand(amount,3))
        wavelengths = np.repeat(300.0,amount)
        gun = Photons(pos,drc,pol,wavelengths)

    sim, analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg.config_name),useGeant4=g4)
    a = perf_resolution(sim,analyzer,gun,42)
    perf_points = a.get_points()
    lns = a.tracks().hit_pos
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(perf_points[:,0],perf_points[:,1],perf_points[:,2])
    ax.scatter(lns.T[:,0],lns.T[:,1],lns.T[:,2])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ix', help='dictionary index')
    args = parser.parse_args()
    ix = int(args.ix)
    detector_list = detectorconfig.DetectorConfigurationList()
    dct = detector_list._get_dict()
    cfg = dct[dct.keys()[ix]]
    #dataset_calibrate(cfg)
    #print 'simulation finished, calibrating...'
    write_perf_lens_calibration(cfg)
