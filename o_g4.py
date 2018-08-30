from detectorconfig import DetectorConfigurationList as dtl
from map_angle_position import perf_resolution as pr
import h5py,time,argparse,os,itertools
from chroma.generator import vertex
import utilities
import paths


def gen_ev(sample,cfg,particle,energy,i_r,o_r):
    seed_loc = 'r%i-%i'%(i_r,o_r)
    data_file_dir = paths.get_data_file_path(cfg.config_name)

    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    fname = data_file_dir+seed_loc+'_'+str(energy)+particle+'_'+'sim.h5'
    sim,analyzer = utilities.sim_setup(cfg, paths.get_calibration_file_name(cfg.config_name), useGeant4=True)

    print 'configuration loaded'

    location = utilities.sph_scatter(sample,i_r*1000,o_r*1000)
    arr_tr, arr_depo = [],[]

    with h5py.File(fname,'w') as f:
        first = True

        for lg in location:
            gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
            perfect_res = pr(sim, analyzer, gun, fov=42.0)
            vert = perfect_res.vert
            tracks = perfect_res.perf_lens_tracks()
            #for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
            #    vert = ev.photons_beg.pos
            #    tracks = analyzer.generate_tracks(ev,qe=(1./3.))

           
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
            first = False

        f.create_dataset('idx_tr',data=arr_tr)
        f.create_dataset('idx_depo',data=arr_depo)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cf', help='detector configuration')
    args = parser.parse_args()
    cf = args.cf
    sample = 500
    energy = 2
    detector_list = dtl()
    config = detector_list.get_configuration(cf)
    for seed in list(itertools.product(['01','34'],['gamma','e-'])):
        start_time = time.time()
        gen_ev(sample,config,seed[1],energy,int(seed[0][0]),int(seed[0][1]))
        print time.time()-start_time
