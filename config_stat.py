import DetectorResponseGaussAngle as dr
import matplotlib.pyplot as plt
import numpy as np

import paths

if __name__ == '__main__':
    for s in ['19','20']:
        arr = []
        cfg = 'cfSam1_%s'%s
        print cfg
        in_file = paths.get_calibration_file_name(cfg)
        det_res = dr.DetectorResponseGaussAngle(cfg,10,10,10,in_file)
        px_lens_means = np.split(det_res.means,20*det_res.n_lens_sys,axis=1)
        px_lens_sigmas = np.split(det_res.sigmas,20*det_res.n_lens_sys)
        for plm,pls in zip(px_lens_means,px_lens_sigmas):
            p_arr = np.einsum('ij,ij->j',plm,np.roll(plm,1,axis=1))[1:]
            p_arr[p_arr>1]=1
            arr.extend(np.arccos(p_arr)*360/6.28)
        if s[0] == 'M':
            bn = 1000
        elif s[0] == 'K':
            bn = 100
        elif s[0] == 'k':
            bn = 50
        plt.hist(arr,bins=1000)
        plt.yscale('log', nonposy='clip')
        plt.show()
