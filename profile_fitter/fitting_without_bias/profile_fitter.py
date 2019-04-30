import numpy as np
import cluster_toolkit as ct
import scipy.optimize as op
import scipy.interpolate as interp
import emcee, os
import aemulus_extras as ae
import aemulus_data as ad

method = "Nelder-Mead"

class SingleSnapshotFitter(object):

    def __init__(self, box, snapshot, fit_in_log=True):
        self.box = box
        self.snapshot = snapshot
        self.data_dict = {}
        self.data_dict["r"] = np.load("datafiles/r.npy")
        self.data_dict["xihm_all"] = \
            np.load("datafiles/xihm_Box%d_Z%d.npy"%(box, snapshot))
        self.data_dict["covs_all"] = \
            np.load("datafiles/covs_Box%d_Z%d.npy"%(box, snapshot))
        self.data_dict["masses"] = \
            np.load("datafiles/masses_Box%d_Z%d.npy"%(box, snapshot))
        self.extras = ae.Extras(box)
        self.z = 1./ad.scale_factors()[snapshot] - 1.
        cosmology = ad.building_box_cosmologies()[box]
        self.cosmology = cosmology
        self.h = cosmology[5]/100.
        self.Omega_m = cosmology[1]/self.h**2
        self.fit_in_log = fit_in_log

    def setup_fitting_args(self, mass_index):
        M = self.data_dict["masses"][mass_index]
        r = self.data_dict["r"]
        xihm = self.data_dict["xihm_all"][mass_index]
        cov = self.data_dict["covs_all"][mass_index]
        icov = np.linalg.inv(cov)

        #Pull out all pre-computed curves from the Extras object
        Extra = self.extras
        k = Extra.k
        P = Extra.P_lin[self.snapshot]
        r_extra = Extra.r
        xi_nl_extra = Extra.xi_nl[self.snapshot]
        xi_nl_spl = interp.InterpolatedUnivariateSpline(r_extra, xi_nl_extra)
        xi_nl = xi_nl_spl(r) #xi_nl evaluated at our data

        #xi_nl = Extra.xi_mm[self.snapshot]
        
        Marr = Extra.M
        lnMarr = np.log(Marr)
        biases = Extra.bias[self.snapshot]
        bias_spline = interp.InterpolatedUnivariateSpline(lnMarr, biases)
        bias = bias_spline(np.log(M))
        h = self.cosmology[5]/100.
        Omega_b = self.cosmology[0]/h**2
        Omega_m = self.cosmology[1]/h**2
        ns = self.cosmology[3]

        #Assemble an args dictionary
        args = {"r":np.ascontiguousarray(r), "xihm":xihm,
                "cov":cov, "icov":icov,
                "xi_nl":np.ascontiguousarray(xi_nl),
                "bias":bias}
        return args
        
    def lnlike(self, params, M, r, xihm, icov, args):
        if self.fit_in_log:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex = \
                np.exp(params)
        else:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex = \
                params

        #Prior is implemented here
        if conc < .1 or alpha < .01 or rt < 0.1 or r_eff < 0.1:
            print("first prior violated")
            return -1e99
        if beta < 0.01 or beta_eff < 0.01 or beta_ex < 0.01:
            print("second prior violated")
            return -1e99
        if r_A < r_B or r_B < 0.1 or r_A > 10:
            print("in here", r_A, r_B)
            return -1e-99
        
        bias = args["bias"]
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m
        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha, rt, beta,
                                                  r_eff, beta_eff,
                                                  r_A, r_B, beta_ex, bias,
                                                  xi_nl, Omega_m)
        X = xihm - xi_ex
        return -0.5*np.dot(X, np.dot(icov, X))
        
    def perform_best_fit(self, mass_index, quiet=False,
                         bfoutpath=None, ihessoutpath=None):
        if bfoutpath is None:
            bfoutpath = "bestfit_values_Box%d_Z%d"%(self.box, self.snapshot)
        if ihessoutpath is None:
            ihessoutpath = "invhessian_Box%d_Z%d"%(self.box, self.snapshot)

        M = self.data_dict["masses"][mass_index]
        r = self.data_dict["r"]
        xihm = self.data_dict["xihm_all"][mass_index]
        cov = self.data_dict["covs_all"][mass_index]
        icov = np.linalg.inv(cov)
        args = self.setup_fitting_args(mass_index)
        #nu_lnM_spline = args["nu_lnM_spline"]
            
        def lnprob(pars, M, r, xihm, icov, args):
            return self.lnlike(pars, M, r, xihm, icov, args)

        #Parameters are log of:conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex
        if self.fit_in_log:
            guess = np.log(np.array([6.5, #conc
                                     0.18, #alpha
                                     1.8, #rt
                                     0.25, #beta
                                     1.6, #r_eff
                                     0.7, #beta_eff
                                     1.8, #r_A
                                     0.6, #r_B
                                     0.28, #beta_ex
            ]))
        else:
            guess = np.array([6.5, #conc
                              0.18, #alpha
                              1.8, #rt
                              0.25, #beta
                              1.6, #r_eff
                              0.7, #beta_eff
                              1.8, #r_A
                              0.6, #r_B
                              0.28, #beta_ex
            ])
        result = op.minimize(lambda *args: -lnprob(*args),
                             guess, args=(M,r,xihm,icov,args),
                             method=method)
        if not quiet:
            print result

        if not os.path.isfile(bfoutpath+".npy"):
            NM = len(self.data_dict["masses"])
            Np = len(guess)
            bfs = np.zeros((NM, Np))
        else:
            bfs = np.load(bfoutpath+".npy")
        bfs[mass_index] = result.x
        np.save(bfoutpath, bfs)
        #np.save(ihessoutpath, result.hess_inv)
        return

    def get_bestfit_model(self, mass_index, bfoutpath=None):
        if bfoutpath is None:
            bfoutpath = "bestfit_values_Box%d_Z%d.npy"%(self.box, self.snapshot)
        if self.fit_in_log:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex = \
                np.exp(np.load(bfoutpath)[mass_index])
        else:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex = \
                np.load(bfoutpath)[mass_index]
        args = self.setup_fitting_args(mass_index)
        M = self.data_dict["masses"][mass_index]
        Extra = self.extras
        #r = Extra.r
        r = args["r"]
        xi_nl = args["xi_nl"]
        bias = args["bias"]
        Omega_m = self.Omega_m

        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha, rt, beta,
                                                  r_eff, beta_eff,
                                                  r_A, r_B, beta_ex, bias,
                                                  xi_nl, Omega_m)
        #xi_ex = args["xi_nl"] * args["bias"]
        return r, xi_ex
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for zi in range(9, 10):
        f = SingleSnapshotFitter(0, zi, True)
        NM = len(f.data_dict["masses"])
        MI = 20
        for i in range(MI, MI+1):
#        for i in range(NM):
            f.perform_best_fit(i)
            fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
            r,xi = f.get_bestfit_model(i)
            ax[0].loglog(r,xi,zorder=-1,c='b')
            rd = f.data_dict["r"]
            xihm = f.data_dict["xihm_all"][i]
            cov = f.data_dict["covs_all"][i]
            err = np.sqrt(cov.diagonal())
            ax[0].errorbar(rd, xihm, err, c='k', zorder=0)
            xim_spl = interp.InterpolatedUnivariateSpline(r, xi)
            xim = xim_spl(rd) #model at the data radii
            ax[1].errorbar(rd, 100*(xihm-xim)/xim, 100*err/xim, c='b', ls='')
            ax[1].axhline(0, ls='--', c='k')
            yl = 20
            ax[1].set_ylim((-yl, yl))
            #ax[1].set_xlim((rd[0], rd[-1]))
            ax[0].set_title("box%d Z%d M%d"%(0,zi,i))
            fig.savefig("figs/box%d_Z%d_M%d.png"%(0,zi,i))
            plt.show()
            plt.cla()
            plt.clf()
            plt.close()
