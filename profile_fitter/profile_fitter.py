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
        self.extras = ae.Extras(box)
        self.redshifts = 1./ad.scale_factors() - 1.
        
    def lnlike(self, params, M, r, xihm, icov):
        #Also has the prior
        conc, rt, beta, nua, nub = np.exp(params)
        if nua < nub:
            return -1e-99
        Ma = self.M_nu_spline(nua)
        Mb = self.M_nu_spline(nub)
        ca = self.c_nu_spline(nua)
        cb = self.c_nu_spline(nub)
        bias = self.bias
        xi_nl = self.xi_nl
        Omega_m = self.Omega_m
        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, rt, beta, Ma, ca,
                                                  Mb, cb, bias, xi_nl, Omega_m,
                                                  exclusion_scheme=0)
        X = xihm - xi_ex
        return -0.5*np.dot(X, np.dot(icov, X))

    
        
    def perform_best_fit(self, quiet=False):
        
        
if __name__ == "__main__":
    f = fitter()
