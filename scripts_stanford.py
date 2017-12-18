from ShortIO.root_short import ShortRootWriter
from mpl_toolkits.mplot3d import Axes3D
from chroma.detector import Detector
from chroma.generator import vertex
from chroma.sim import Simulation
import matplotlib.pyplot as plt
import detectoranalysis as da
import lensmaterials as lm
from chroma import sample
import kabamland2 as kb
import argparse, os
import paths


datadir = paths.detector_calibration_path

parser = argparse.ArgumentParser()
parser.add_argument('cfg', help='configuration')
parser.add_argument('fnc', help='full_detector or detres')
args = parser.parse_args()

fileinfo = args.cfg

if args.fnc == 'full_detector':
	kb.full_detector_simulation(100000, fileinfo, 'sim-'+fileinfo+'_100million.root', datadir=datadir)

elif args.fnc == 'detres':
	da.create_detres(fileinfo, 'sim-'+fileinfo+'_100million.root', 'detresang-'+fileinfo+'_1DVariance_100million.root', method="GaussAngle", nevents=-1, datadir=datadir)
	os.remove(datadir+'sim-'+fileinfo+'_100million.root')
