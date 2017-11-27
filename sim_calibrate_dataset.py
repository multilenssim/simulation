import os, itertools

l_base = [1,8]
EPDR = [8]
configs = ['cfSam1_K%i_%i'%(k[0],k[1]) for k in list(itertools.product(l_base,EPDR))]

for cfg in configs:
	print '----------------------------------------------------------------%s----------------------------------------------------------------'%cfg
	for s_d in ['01','34']: 
		if os.path.exists('/home/jacopodalmasson/Desktop/dev/'+cfg+'/raw_data'):
			print 'simulation part'
	       		os.system('python g4_sim.py e- %s %s'%(s_d,cfg))
	        	os.system('python g4_sim.py gamma %s %s'%(s_d,cfg))

		else:
			print 'setting up the detector'
		        os.system('python scripts_stanford.py '+cfg+' full_detector')
		        os.system('python scripts_stanford.py '+cfg+' detres')
		        os.system('python save_conf.py '+cfg)
			print 'simulation part'
                        os.system('python g4_sim.py e- %s %s'%(s_d,cfg))
                        os.system('python g4_sim.py gamma %s %s'%(s_d,cfg))
