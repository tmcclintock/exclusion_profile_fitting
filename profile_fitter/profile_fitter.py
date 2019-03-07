import numpy as np
import cluster_toolkit as ct
import scipy.optimize as op
import emcee
import aemulus_extras as ae
import aemulus_data as ad

class SingleSnapshotFitter(object):

    def __init__(self, box, snapshot):
        self.box = box
        self.snapshot = snapshot
        self.data_dict = {}
        self.data_dict["r"] = np.load("datafiles/r.npy")
        self.data_dict["xihm_all"] = \
            np.load("datafiles/xihm_Box%d_Z%d.npy"%(box, snapshot))
        self.dict_data["covs_all"] = \
            np.load("covs_Box%d_Z%d.npy"%(box, snapshot))
        self.data_dict["masses"] = \
            np.load("datafiles/masses_Box%d_Z%d.npy"%(box, snapshot))

    def lnlike(self, params, r, xihm, icov):
        
        
    def perform_best_fit(self, quiet=False):
        
        
if __name__ == "__main__":
    f = fitter()
