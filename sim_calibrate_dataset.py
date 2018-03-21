import os, itertools
import paths

l_base = [200]
EPDR = [8]
configs = ['cfSam1_K%i_%i_small'%(k[0],k[1]) for k in list(itertools.product(l_base,EPDR))]
cb = '_narrow'

for cfg in configs:
	print '----------------------------------------------------------------%s----------------------------------------------------------------'%cfg
	print 'setting up the detector'
	os.system('python scripts_stanford.py %s full_detector %s'%(cfg,cb))
	os.system('python scripts_stanford.py %s detres %s'%(cfg,cb))
	if os.path.exists('%s%s.pickle'%(paths.detector_pickled_path,cfg)):
		pass
	else:
		print 'saving configuration'
	        os.system('python save_conf.py '+cfg)
