import matplotlib.pyplot as plt
import DetectorResponse as dr
import matplotlib.cm as cm
import detectorconfig
import numpy as np
import h5py as h5
import paths

def normalize(arr, ax):
    return np.einsum('ij,i->ij',arr,1/np.linalg.norm(arr,axis=ax))

def proj(cfg):
    #with h5.File('%sno_perf_cal_h5fied_%s.h5'%(paths.detector_calibration_path,cfg.config_name),'r') as f:
    #with h5.File('%sh5fied.h5'%paths.detector_calibration_path,'r') as f:
    #with h5.File('%sno_perf_cal_%s.h5'%(paths.detector_calibration_path,cfg.config_name),'r') as f:
    with h5.File('%sdetresang-%s_1DVariance_100million.h5'%(paths.detector_calibration_path,cfg.config_name),'r') as f:
            means = f['means'][:]

    det_res = dr.DetectorResponse(cfg)
    means = means.T
    n_lens = det_res.n_lens_sys
    n_pmts_per_surf = det_res.n_pmts_per_surf
    lns_center = det_res.lens_centers
    px_lens_means = means.reshape((n_lens,n_pmts_per_surf,3))
    u_dir = np.cross(lns_center,[0,0,1])
    mask = np.all(u_dir==0,axis=1)
    u_dir[mask] = np.cross(lns_center[mask],[0,1,0])
    u_dir = normalize(u_dir,1)
    v_dir = normalize(np.cross(lns_center, u_dir),1)
    u_proj = np.einsum('ijk,ik->ij',px_lens_means,u_dir)
    v_proj = np.einsum('ijk,ik->ij',px_lens_means,v_dir)
    return px_lens_means, u_proj, v_proj


if __name__ == '__main__':
    cut = False
    detector_list = detectorconfig.DetectorConfigurationList()
    dct = detector_list._get_dict()
    cfg = dct[dct.keys()[7]]
    px_lens_means, u_proj, v_proj = proj(cfg)
    sin_dir = np.linalg.norm([u_proj.flatten(),v_proj.flatten()],axis=0)
    _,bn,_ = plt.hist(np.arcsin(sin_dir),bins=100)
    #asn = np.arcsin(sin_dir).reshape((u_proj.shape))
    plt.yscale('log', nonposy='clip')
    plt.xlabel('calibrated pixel angular aperture')
    plt.show()

    if cut:
        plt.hist(px_lens_sigmas.flat,bins=100,color='b')
        plt.yscale('log', nonposy='clip')
        plt.xlabel('sigma value')
        plt.show()
        cut_val = np.fromstring(raw_input('cut value: '),dtype=float,sep=' ')
        colors = cm.rainbow(np.linspace(0,1,len(cut_val)))

    for i in np.random.choice(u_proj.shape[0],4,replace=False):
        circle = plt.Circle((0,0),1,facecolor='none',edgecolor='r')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)
        ax.scatter(u_proj[i],v_proj[i],c='b')

        if cut:
            for cv,cl in reversed(zip(cut_val,colors)):
                ax.scatter(u_proj[i,px_lens_sigmas[i]<cv],v_proj[i,px_lens_sigmas[i]<cv],c=cl,s=50)

        ax.set_xlim(-0.55,0.55)
        ax.set_ylim(-0.55,0.55)
        ax.set_aspect('equal')
        plt.show()
