from chroma.sim import Simulation
from chroma import sample
from chroma.generator import vertex
from ShortIO.root_short import ShortRootWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from chroma.detector import Detector
import lensmaterials as lm
import argparse, os
import kabamland2 as kb
import detectoranalysis as da

datadir = "/home/miladmalek/TestData/"

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
