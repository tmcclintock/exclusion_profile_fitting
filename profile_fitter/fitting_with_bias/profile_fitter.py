import numpy as np
import cluster_toolkit as ct
import scipy.optimize as op
import scipy.interpolate as interp
import emcee, os
import aemulus_extras as ae
import aemulus_data as ad

method = "Nelder-Mead"
datapath = "/Users/tmcclintock/Data/aemulus_hmcfs/simplified_data/"

class SingleSnapshotFitter(object):

    def __init__(self, box, snapshot, fit_in_log=True):
        self.box = box
        self.snapshot = snapshot
        self.data_dict = {}
        self.data_dict["r"] = np.load("../datafiles/r.npy")
        self.data_dict["xihm_all"] = \
            np.load(datapath+"xihm_Box%d_Z%d.npy"%(box, snapshot))
        self.data_dict["covs_all"] = \
            np.load(datapath+"covs_Box%d_Z%d.npy"%(box, snapshot))
        self.data_dict["masses"] = \
            np.load(datapath+"masses_Box%d_Z%d.npy"%(box, snapshot))
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
        h = self.cosmology[5]/100.
        Omega_b = self.cosmology[0]/h**2
        Omega_m = self.cosmology[1]/h**2
        ns = self.cosmology[3]

        #Assemble an args dictionary
        args = {"r":np.ascontiguousarray(r), "xihm":xihm,
                "cov":cov, "icov":icov,
                "xi_nl":np.ascontiguousarray(xi_nl)}
        return args
        
    def lnlike(self, params, M, r, xihm, icov, args):
        if self.fit_in_log:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex, bias = \
                np.exp(params)
        else:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex, bias = \
                params
        #print params
        #print conc, alpha, rt, beta
        #Prior is implemented here
        quiet = True
        if conc < .1 or conc > 20:
            if not quiet:
                print("first prior violated", conc)
            return -1e99
        if alpha < .04 or alpha > 3 or rt < 0.1 or r_eff < 0.1:
            if not quiet:
                print("here")
            return -1e99
        if beta < 0.01 or beta_eff < 0.01 or beta_ex < 0.01:
            if not quiet:
                print("second prior violated")
            return -1e99
        if r_A < r_B or r_B < 0.1 or r_A > 10:
            if not quiet:
                print("r_A r_B prior violated", r_A, r_B)
            return -1e99
        
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m
        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha, rt, beta,
                                                  r_eff, beta_eff,
                                                  r_A, r_B, beta_ex, bias,
                                                  xi_nl, Omega_m)
        X = xihm - xi_ex
        return -0.5*np.dot(X, np.dot(icov, X))
        
    def perform_best_fit(self, mass_index, quiet=False,
                         bfoutpath=None):
        if bfoutpath is None:
            bfoutpath = "bestfit_values_Box%d_Z%d"%(self.box, self.snapshot)

        M = self.data_dict["masses"][mass_index]
        r = self.data_dict["r"]
        xihm = self.data_dict["xihm_all"][mass_index]
        cov = self.data_dict["covs_all"][mass_index]
        icov = np.linalg.inv(cov)
        args = self.setup_fitting_args(mass_index)
            
        def lnprob(pars, M, r, xihm, icov, args):
            return self.lnlike(pars, M, r, xihm, icov, args)

        #Parameters
        guess = np.array([6.5, #conc
                          0.18, #alpha
                          1.8, #rt
                          0.25, #beta
                          1.6, #r_eff
                          0.7, #beta_eff
                          1.8, #r_A
                          0.6, #r_B
                          0.28, #beta_ex
                          2.5, #bias
        ])
        if self.fit_in_log:
            #guess = np.log(guess)
            guess = np.array([2.62, -2.4, -0.0252, -0.75,  0.35,
                              0.0117,  0.053 , -1.86, -0.863,  0.408])
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
        return

    def perform_mcmc(self, mass_index, quiet=False,
                         bfoutpath=None, chainpath=None, likespath=None):

        if chainpath is None:
            chainpath = "chains/chain_Box{0}_Z{1}".format(self.box, \
                                                          self.snapshot)
        if likespath is None:
            likespath = "chains/likes_Box{0}_Z{1}".format(self.box, \
                                                          self.snapshot)
        if bfoutpath is None:
            bfoutpath = "bestfit_values_Box%d_Z%d.npy"%(self.box, self.snapshot)
        print("Best params:", np.load(bfoutpath)[mass_index])
        params = np.load(bfoutpath)[mass_index]

        M = self.data_dict["masses"][mass_index]
        r = self.data_dict["r"]
        xihm = self.data_dict["xihm_all"][mass_index]
        cov = self.data_dict["covs_all"][mass_index]
        icov = np.linalg.inv(cov)
        args = self.setup_fitting_args(mass_index)

        ndim, nwalkers = len(params), 40
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnlike,\
                                        args=(M,r,xihm,icov,args))
        print("Running first burn-in")
        p0 = params + np.fabs(params)*1e-4*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 100)
        #print lp.shape
        #print lp
        #exit()
        print("Running second burn-in")
        p0 = p0[np.argmax(lp)] + \
            np.fabs(params)*1e-4*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        print("Running production...")
        sampler.run_mcmc(p0, 2000);
        chain = sampler.flatchain
        print chain.shape
        print np.mean(chain,0)
        lnlikes = sampler.flatlnprobability
        np.save(chainpath, chain)
        np.save(likespath, lnlikes)
        print("MCMC finished for Box{0} Z{1}.".format(self.box, self.snapshot))
        return

    def get_bestfit_model(self, mass_index, bfoutpath=None):
        if bfoutpath is None:
            bfoutpath = "bestfit_values_Box%d_Z%d.npy"%(self.box, self.snapshot)
        if self.fit_in_log:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex, bias = \
                np.exp(np.load(bfoutpath)[mass_index])
        else:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex, bias = \
                np.load(bfoutpath)[mass_index]
        print("Best params:", np.load(bfoutpath)[mass_index])
        args = self.setup_fitting_args(mass_index)
        M = self.data_dict["masses"][mass_index]
        Extra = self.extras
        #r = Extra.r
        r = args["r"]
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m

        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha, rt, beta,
                                                  r_eff, beta_eff,
                                                  r_A, r_B, beta_ex, bias,
                                                  xi_nl, Omega_m)
        return r, xi_ex

    def get_MAP_model(self, mass_index, chainpath=None, likespath=None):
        if chainpath is None:
            chainpath = "chains/chain_Box{0}_Z{1}.npy".format(self.box, \
                                                              self.snapshot)
        if likespath is None:
            likespath = "chains/likes_Box{0}_Z{1}.npy".format(self.box, \
                                                              self.snapshot)
        lnlikes = np.load(likespath)
        chain = np.load(chainpath)

        if self.fit_in_log:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex, bias = \
                np.exp(chain[np.argmax(lnlikes)])
        else:
            conc, alpha, rt, beta, r_eff, beta_eff, r_A, r_B, beta_ex, bias = \
                chain[np.argmax(lnlikes)]
        print("MAP params:", chain[np.argmax(lnlikes)])
        args = self.setup_fitting_args(mass_index)
        M = self.data_dict["masses"][mass_index]
        Extra = self.extras
        #r = Extra.r
        r = args["r"]
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m

        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha, rt, beta,
                                                  r_eff, beta_eff,
                                                  r_A, r_B, beta_ex, bias,
                                                  xi_nl, Omega_m)
        return r, xi_ex

        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rc("text", usetex=True)
    plt.rc("font", size=24, family="serif")
    plt.rc("errorbar", capsize=2)
    box = 0
    for box in range(box, box+1):
        for zi in range(9,10):
            f = SingleSnapshotFitter(box, zi, fit_in_log=False)
            NM = len(f.data_dict["masses"])
            MI = 10
            for i in range(MI, MI+1):
            #for i in range(NM):
                #f.perform_best_fit(i)
                #f.perform_mcmc(i)

                fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
                plt.subplots_adjust(hspace=0)
                #r,xi = f.get_bestfit_model(i)
                r,xi = f.get_MAP_model(i)
                ax[0].loglog(r,xi,zorder=-1,c='b')
                rd = f.data_dict["r"]
                xihm = f.data_dict["xihm_all"][i]
                cov = f.data_dict["covs_all"][i]
                err = np.sqrt(cov.diagonal())
                ax[0].errorbar(rd, xihm, err, c='k', marker='.', zorder=0)
                xim_spl = interp.InterpolatedUnivariateSpline(r, xi)
                xim = xim_spl(rd) #model at the data radii
                ax[1].errorbar(rd, 100*(xihm-xim)/xim, 100*err/xim, c='b',
                               marker='.', ls='')
                ax[1].axhline(0, ls='--', c='k')
                #ax[0].set_yticks([.01, .1, 1, 10, 100, 1000])
                #ax[0].set_yticklabels([.1, 1, 10, 100, 1000])
                ax[0].set_ylabel(r"$\xi_{\rm hm}(r,M)$")
                ax[1].set_ylabel(r"$\Delta\ [\%]$")
                ax[1].set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")

                yl = 21
                ax[1].set_ylim((-yl, yl))
                #ax[1].set_xlim((rd[0], rd[-1]))
                #ax[0].set_title("box%d Z%d M%d"%(box,zi,i))
                M = f.data_dict["masses"][i]
                z = f.z
                ax[0].text(.09, .03, "$M=4e13\ [h^{-1}\mathrm{M_\odot}]$", transform=ax[0].transAxes)
                ax[0].text(.09, 0.25, "$z=%.0f$"%(z), transform=ax[0].transAxes)
                #ax[0].set_title(r"$M=%.1e$"%(z,M))

                fig.savefig("figs/box%d_Z%d_M%d.png"%(box,zi,i), dpi=300, bbox_inches="tight")
                plt.show()
                plt.cla()
                plt.clf()
                plt.close()
