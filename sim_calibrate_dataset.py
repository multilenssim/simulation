import os, argparse

import kabamland2 as kb
import detectoranalysis as da
import lensmaterials
import detectorconfig
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('cfg', help='configuration')
args = parser.parse_args()
cfg = args.cfg

datadir = '/home/ubuntu/Development/TestData/'
photons_file = 'sim-'+cfg+'_100million.root'

if not os.path.exists(datadir+cfg):   # This is not a great structure as other configuration data may change in addition to the detector config
        # We should really date stamp the directory containing the output and configuration files
	print '==== setting up the detector ===='
        if not os.path.exists(datadir + photons_file):
                kb.full_detector_simulation(100000, cfg, photons_file, datadir=datadir)
        print("==== Detector built  ====")
        da.create_detres(args.cfg, photons_file, 'detresang-'+cfg+'_1DVariance_100million.root', method="GaussAngle", nevents=1000, datadir=datadir)
        #os.remove(photons_file)
        print("==== Calibration complete ====")

if True:
        config_path = datadir+cfg+'/raw_data/'
        if not os.path.exists(config_path):
                os.makedirs(config_path)
        all_config_info = {'configuration': detectorconfig.configdict[cfg].__dict__}
        all_config_info['scintillator'] = lensmaterials.create_scintillation_material().__dict__
        all_config_info['lens_material'] = lensmaterials.lensmat.__dict__
        all_config_info['G4_config'] = 'placeholder'
        all_config_info['particle_config'] = 'placeholder'
        with open(config_path+'full_config.pickle', 'w') as outf:
                pickle.dump(all_config_info, outf)

        # Write both files for now to support Jacopo's test setup
        with open(config_path+'conf.pkl', 'w') as outf:
                pickle.dump(detectorconfig.configdict[cfg].__dict__, outf)

print 'simulation part'
os.system('python g4_sim.py e- '+cfg)
os.system('python g4_sim.py gamma '+cfg)
