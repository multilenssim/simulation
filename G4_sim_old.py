from chroma.generator import vertex
import matplotlib.pyplot as plt
import argparse, h5py
import detectorconfig
from paths import *
import numpy as np
import config_stat
import utilities

def k_thresh(r_idx,m=0.511):
# return the Cerenkov threshold energy (kinetic) for a given particle (mass in MeV) crossing media with r_idx as refractive index
    s = r_idx*r_idx
    return m*(1/np.sqrt(1.0-1.0/s)-1.0)


def sim_test(sim):
    direction      = np.array([1.0,1.0,1.0])
    starting_point = np.cross(np.random.rand(3),direction)
    gun            = vertex.particle_gun(['neutron'], vertex.constant(starting_point), vertex.constant(direction), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))    
    for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
        vtx = ev.photons_beg.track_tree

    return vtx


def write_neutrons():
    config = detectorconfig.get_detector_config(cfg)
    sim,analyzer = utilities.sim_setup(config, get_calibration_file_name(config.config_name), useGeant4=True)
    time     = np.zeros((amount,n_scattering))
    particle = np.chararray((amount,n_scattering),itemsize=3)
    position = np.zeros((amount,n_scattering,3))
    photon   = np.zeros((amount,n_scattering))
    energy   = np.zeros((amount,n_scattering))
    n_en     = []

    for i in xrange(amount):
        tm  = []
        pos = []
        prt = []
        en  = []
        ph  = []
        vtx = sim_test(sim)

        for key, value in vtx.iteritems():
            try:
                en.append(value['energy'])
                ph.append(value['child_processes']['Scintillation'])
                tm.append(value['time'])
                pos.append(value['position'])
                prt.append(value['particle'])

            except KeyError: pass

        first_second_index = np.argsort(tm)[1:n_scattering+1]

        try:
            if tm[first_second_index[1]]<100:
                photon[i]      = [ph[first_second_index[k]] for k in xrange(n_scattering)]
                energy[i]   = [en[first_second_index[k]] for k in xrange(n_scattering)]
                position[i] = [pos[first_second_index[k]] for k in xrange(n_scattering)]
                time[i]     = [tm[first_second_index[k]] for k in xrange(n_scattering)]
                particle[i] = [prt[first_second_index[k]] for k in xrange(n_scattering)]

            else:
                print [prt[k] for k in xrange(3)], [en[k] for k in xrange(3)]
                print en[0]-en[first_second_index[0]]
                n_en.append(en[0]-en[first_second_index[0]])

        except IndexError: pass

    mask     = np.where(photon == np.zeros(n_scattering))[0]
    photon   = np.delete(photon,mask,axis=0)
    energy   = np.delete(energy,mask,axis=0)
    position = np.delete(position,mask,axis=0)
    time     = np.delete(time,mask,axis=0)
    particle = np.delete(particle,mask,axis=0)

    with h5py.File(fname,'w') as f:
       f.create_dataset('position',maxshape=position.shape,data=position,chunks=True)
       f.create_dataset('time',maxshape=time.shape,data=time,chunks=True)
       f.create_dataset('particle',maxshape=particle.shape,data=particle,chunks=True)
       f.create_dataset('energy',maxshape=energy.shape,data=energy,chunks=True)
       f.create_dataset('photon',maxshape=photon.shape,data=photon,chunks=True)


def read_neutrons():
    global pos, time, part, energy, photon, en_scattering

    with h5py.File(fname,'r') as f:
        pos    = f['position'][:]
        time   = f['time'][:]
        part   = f['particle'][:]
        energy = f['energy'][:]
        photon = f['photon'][:]


    mask   = np.where(part[:,1]=='gam')
    photon = np.delete(photon,mask,axis=0)
    energy = np.delete(energy,mask,axis=0)
    pos    = np.delete(pos,mask,axis=0)
    time   = np.delete(time,mask,axis=0)
    part = np.delete(part,mask,axis=0)
    photon[np.logical_and(part!='pro',part!='C12')] = 0
    energy[np.logical_and(part!='pro',part!='C12')] = 0 
    en_scattering = np.argsort(photon,axis=1)


def cos_comparison():
    initial_dir   = config_stat.normalize(pos[:,0],1)
    scattered_dir = config_stat.normalize(pos[:,1]-pos[:,0],1)
    expected_cos  = np.einsum('ij,ij->i',initial_dir,scattered_dir)
    kinematic_cos = np.sqrt(1-energy[range(photon.shape[0]),en_scattering[:,-1]]/2)
    expected_cos = expected_cos[~np.isnan(kinematic_cos)]
    kinematic_cos = kinematic_cos[~np.isnan(kinematic_cos)]
    plt.hist(kinematic_cos-expected_cos,bins=50)
    plt.show()


def dist():
    mask = np.logical_or(part[:,0]=='pro',part[:,0]=='C12')
    dist         = np.linalg.norm(pos[mask,0] - pos[mask,1],axis=1)
    dist_neutron = np.linalg.norm(pos[part[:,0]=='pro',0] - pos[part[:,0]=='pro',1],axis=1)
    fig,ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.hist(dist,bins=50,histtype='step',density=1,color='blue')
    #ax2.hist(dist,bins=50,histtype='step',cumulative=1,density=1,color='red')
    ax1.hist(dist_neutron,bins=50,histtype='step',density=1,color='red')
    plt.show()


def timing():
    mask = np.logical_or(part[:,0]=='pro',part[:,0]=='C12')
    #dist = time[mask,1] - time[mask,0]
    dist = np.linalg.norm(pos[:,0],axis=1)
    fig,ax1 = plt.subplots()
    ax1.hist(dist,bins=50,histtype='step',cumulative=-1,color='blue')
    ax1.set_xlabel('distance before first scattering - cumulative distribution - [mm]')
    plt.show()


def most_energetic_scattering():
    en_scattering = en_scattering[:,-1]
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist(en_scattering[part[:,0]=='pro'],bins=np.linspace(0,max(en_scattering),max(en_scattering)+1),histtype='step',density=1,color='blue')
    ax2.hist(en_scattering[part[:,0]=='pro'],bins=np.linspace(0,max(en_scattering),max(en_scattering)+1),histtype='step',cumulative=1,density=1,color='red')
    ax1.set_xlabel('most energetic scattering')
    ax1.set_ylabel('counts',color='blue')
    ax2.set_ylabel('cumulative',color='red')
    plt.show()


def blurred_reconstruction(plot):
    import sklearn.cluster as clust

    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.colors as cl
        import matplotlib.cm as cm

    d_res      = 20
    min_photon = 15

    for i,tp in enumerate(photon[50:60]):
        mask = tp!=0
        #clust_pos = pos[i,mask],tp[mask],np.repeat(pos[i,mask],tp[mask],axis=0)
        db = clust.DBSCAN(eps=d_res,min_samples=min_photon,n_jobs=4)
        db.fit(pos[i,mask],sample_weight=tp[mask])
        cluster   = np.unique(db.labels_)
        avg_pos   = np.zeros((len(cluster),3))
        tot_light = np.zeros(len(cluster))

        for idx,lb in enumerate(cluster):
            avg_pos[idx] = np.average(pos[i,mask][db.labels_==lb],axis=0,weights=tp[mask][db.labels_==lb])
            tot_light[idx] = sum(tp[mask][db.labels_==lb])

        if plot:
            plot_blobs(cm,cl,db,i,tp,mask) 


def plot_blobs(cm,cl,db,i,tp,mask):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cmap = cm.get_cmap('gist_rainbow')
    normalize = cl.Normalize(vmin=min(db.labels_), vmax=max(db.labels_))
    colors = [cmap(normalize(value)) for value in db.labels_]
    for ix,ps in enumerate(pos[i,mask]):
        ax.scatter(ps[0],ps[1],ps[2],s=int(tp[mask][ix]),c=colors[ix])
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='detector configuration')
    args = parser.parse_args()
    cfg = args.cfg
    amount = 10000
    energy = 2
    n_scattering = 8
    fname = '%sneutron_data.h5'%data_files_path
    #write_neutrons()
    read_neutrons()
    print 'neutron captures faster than 100ns: %i'%sum(np.logical_not(np.logical_or(np.logical_or(part[:,1]=='pro',part[:,1]=='C12'),part[:,1]=='C13')))
    print 'probability of scattering not on a proton: %.2f'%(sum(part[:,0]!='pro')/float(part.shape[0]))
    print 'good events %i'%len(photon)
    #timing()
    blurred_reconstruction(True)
