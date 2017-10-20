import os

configs = ['cfSam1_k%i_8'%k for k in [1,2,3,4,6]]

for cfg in configs:
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
