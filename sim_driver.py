import g4_sim
from multiprocessing import Pool, TimeoutError
import multiprocessing
import subprocess

if __name__=='__main__':
    pool = Pool(multiprocessing.cpu_count())

    #g4_sim.run_simulation('cfSam1_M10_10', 'e-', '01')
    '''
    pool.apply_async(g4_sim.run_simulation, ('cfSam1_M10_8', 'e-', '01'))
    pool.apply_async(g4_sim.run_simulation, ('cfSam1_M10_8', 'gamma', '01'))

    pool.close()
    pool.join()

    '''
    configs = ['cfSam1_M10_10', 'cfSam1_M10_8',
               'cfSam1_M4_10',  'cfSam1_M4_8',
               'cfSam1_M20_10', 'cfSam1_M20_8',
               'cfSam1_K12_10', 'cfSam1_K12_8',
               'cfSam1_K10_10', 'cfSam1_K10_8']

    for config in configs:
        #for particle in ['e-','gamma']:
        for distance in ['01','34']:
            print('python sim_calibrate_dataset.py ' + config + ' ' + distance + ' > ' + config+'-'+distance+'.log &')
            #print('python g4_sim.py ' + particle + ' ' + distance + ' ' + config + ' > ' + config+'-'+particle+'-'+distance+'.log &')
                
    # launch async calls:
    '''
    procs = [subprocess.Popen(['python', 'g4_sim.py', 'gamma', '01', config]) for config in configs]
    procs.extend([subprocess.Popen(['python', 'g4_sim.py', 'e-', '01', config]) for config in configs])
    procs.extend([subprocess.Popen(['python', 'g4_sim.py', 'gamma', '34', config]) for config in configs])
    procs.extend([subprocess.Popen(['python', 'g4_sim.py', 'e-', '34', config]) for config in configs])
    # wait.
    for proc in procs:
        proc.wait()
        # check for results:
        if any(proc.returncode != 0 for proc in procs):
            print 'Something failed'
    '''
