import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('cfg', help='configuration')
args = parser.parse_args()
cfg = args.cfg

if os.path.exists('/home/jacopodalmasson/Desktop/dev/'+cfg):
	print 'simulation part'
        os.system('python g4_sim.py e- '+cfg)
        os.system('python g4_sim.py gamma '+cfg)

else:
	print 'setting up the detector'
        os.system('python scripts_stanford.py '+cfg+' full_detector')
        os.system('python scripts_stanford.py '+cfg+' detres')
        os.system('python save_conf.py '+cfg)
	print 'simulation part'
        os.system('python g4_sim.py e- '+cfg)
        os.system('python g4_sim.py gamma '+cfg)
