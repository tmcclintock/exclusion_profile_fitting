import numpy as np
import cluster_toolkit as ct
import scipy.optimize as op
import scipy.interpolate as interp
import emcee, os
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

        Marr = Extra.M
        nu = Extra.nu[self.snapshot]
        lnMarr = np.log(Marr)
        nu_lnM_spline = interp.InterpolatedUnivariateSpline(lnMarr, nu)
        lnM_nu_spline = interp.InterpolatedUnivariateSpline(nu, lnMarr)
        biases = Extra.bias[self.snapshot]
        bias_spline = interp.InterpolatedUnivariateSpline(lnMarr, biases)
        bias = bias_spline(np.log(M))
        h = self.cosmology[5]/100.
        Omega_b = self.cosmology[0]/h**2
        Omega_m = self.cosmology[1]/h**2
        ns = self.cosmology[3]
        concentrations = np.array([ct.concentration.concentration_at_M(Mi, k, P, ns, Omega_b, Omega_m, h, Mass_type="mean") for Mi in Marr])
        c_nu_spline = interp.InterpolatedUnivariateSpline(nu, concentrations)
        c_lnM_spline = interp.InterpolatedUnivariateSpline(lnMarr, concentrations)
        #Assemble an args dictionary
        args = {"r":np.ascontiguousarray(r), "xihm":xihm,
                "cov":cov, "icov":icov,
                "xi_nl":np.ascontiguousarray(xi_nl),
                "numin":nu[0], "numax":nu[-1], "bias":bias,
                "nu_lnM_spline":nu_lnM_spline, "lnM_nu_spline":lnM_nu_spline,
                "c_nu_spline":c_nu_spline, "c_lnM_spline":c_lnM_spline,
                "lnMmin":lnMarr[0], "lnMmax":lnMarr[-1]}
        return args
        
    def lnlike(self, params, M, r, xihm, icov, args):
        #Also has the prior
        conc, rt, beta, nua, nub = np.exp(params)
        if nua < nub:
            return -1e-99
        if nub < args["numin"]:
            return -1e99
        if nua > args["numax"]:
            return -1e99
        c_nu_spline = args["c_nu_spline"]
        lnM_nu_spline = args["lnM_nu_spline"]
        Ma = np.exp(lnM_nu_spline(nua))
        Mb = np.exp(lnM_nu_spline(nub))
        ca = c_nu_spline(nua)
        cb = c_nu_spline(nub)
        """
        conc, rt, beta, Ma, Mb = np.exp(params)
        if any(params[-2:] < args["lnMmin"]):
            return -1e99
        if any(params[-2:] > args["lnMmax"]):
            return -1e99
        if Ma < Mb:
            return -1e99
        c_lnM_spline = args["c_lnM_spline"]
        ca = c_lnM_spline(np.log(Ma))
        cb = c_lnM_spline(np.log(Mb))
        """
        bias = args["bias"]
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m
        #print "%e"%M
        #print conc, rt, beta, bias
        #print "%e"%Ma, ca, "%e"%Mb, cb
        #print Omega_m
        #exit()
        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, rt, beta, Ma, ca,
                                                  Mb, cb, bias, xi_nl, Omega_m,
                                                  exclusion_scheme=0)
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
        nu_lnM_spline = args["nu_lnM_spline"]
            
        def lnprob(pars, M, r, xihm, icov, args):
            return self.lnlike(pars, M, r, xihm, icov, args)

        #Parameters are log of: conc., r_t, beta, nu_a, nu_b
        guess = np.log([8., 2.67, 3.92,
                                 nu_lnM_spline(np.log(M*5.)),
                                 nu_lnM_spline(np.log(M/2.))])
        #guess = np.log([8.06, 2.67, 3.92, M*5, M/2.])
        nll = lambda *args: -lnprob(*args)
        result = op.minimize(nll, guess, args=(M,r,xihm,icov,args),
                             method="Nelder-Mead")
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
        #conc, rt, beta, Ma, Mb = np.exp(np.load(bfoutpath))
        conc, rt, beta, nua, nub = np.exp(np.load(bfoutpath)[mass_index])
        args = self.setup_fitting_args(mass_index)
        M = self.data_dict["masses"][mass_index]
        Extra = self.extras
        r = Extra.r
        xi_nl = Extra.xi_nl[self.snapshot]
        c_lnM_spline = args["c_lnM_spline"]
        
        c_nu_spline = args["c_nu_spline"]
        lnM_nu_spline = args["lnM_nu_spline"]
        Ma = np.exp(lnM_nu_spline(nua))
        Mb = np.exp(lnM_nu_spline(nub))
        ca = c_nu_spline(nua)
        cb = c_nu_spline(nub)
        #ca = c_lnM_spline(np.log(Ma))
        #cb = c_lnM_spline(np.log(Mb))
        bias = args["bias"]
        Omega_m = self.Omega_m

        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, rt, beta, Ma, ca,
                                                  Mb, cb, bias, xi_nl, Omega_m,
                                                  exclusion_scheme=0)
        return r, xi_ex
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for zi in range(9, 10):
        f = SingleSnapshotFitter(0, zi)
        NM = len(f.data_dict["masses"])
        for i in range(0, NM):
            #f.perform_best_fit(i)
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
            ax[1].set_xlim((rd[0], rd[-1]))
            ax[0].set_title("box%d Z%d M%d"%(0,zi,i))
            fig.savefig("figs/box%d_Z%d_M%d.png"%(0,zi,i))
            plt.cla()
            plt.clf()
            plt.close()
