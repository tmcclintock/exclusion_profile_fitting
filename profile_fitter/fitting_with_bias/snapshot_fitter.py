import numpy as np
import cluster_toolkit as ct
import scipy.optimize as op
import scipy.interpolate as interp
import emcee, os
import aemulus_extras as ae
import aemulus_data as ad

method = "Powell"
datapath = "/Users/tmcclintock/Data/aemulus_hmcfs/simplified_data/"

class SnapshotFitter(object):

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
        #Peak height pivot
        self.nu_pivot = 2.

    def setup_fitting_args(self):
        Ms = self.data_dict["masses"]
        r = self.data_dict["r"]
        xihms = self.data_dict["xihm_all"]
        covs = self.data_dict["covs_all"]
        icovs = [np.linalg.inv(cov) for cov in covs]

        #Pull out all pre-computed curves from the Extras object
        Extra = self.extras
        r_extra = Extra.r
        xi_nl_extra = Extra.xi_nl[self.snapshot]
        xi_nl_spl = interp.InterpolatedUnivariateSpline(r_extra, xi_nl_extra)
        xi_nl = np.ascontiguousarray(xi_nl_spl(r))

        Marr = Extra.M
        lnMarr = np.log(Marr)
        #Evaluate nu(M,z) for the masses in our sim
        nuarr = Extra.nu[self.snapshot]
        nuspl = interp.InterpolatedUnivariateSpline(lnMarr, nuarr)
        nus = nuspl(np.log(Ms))

        #Assemble an args dictionary
        args = {"r":np.ascontiguousarray(r), "xihms":xihms,
                "covs":covs, "icovs":icovs,
                "xi_nl":xi_nl, "nus":nus, "Ms":Ms}
        return args

    def lnlike(self, params, args):
        if self.fit_in_log:
            conc0, conc1, alpha0, alpha2, rt0, rt1,\
                beta, r_eff, beta_eff, r_A0, r_A1, r_B0, r_B1, \
                beta_ex, \
                bias0, bias1 = np.exp(params)
        else:
            conc0, conc1, alpha0, alpha2, rt0, rt1,\
                beta, r_eff, beta_eff, r_A0, r_A1, r_B0, r_B1, beta_ex, \
                bias0, bias1 = params
        
        #Priors part 1 here
        if r_eff < 0.1 or beta < 0.01 or beta_eff < 0.01 or beta_ex < 0.01:
            print("in prior 1 ",r_eff, beta, beta_eff, beta_ex)
            return -1e99

        Ms = args["Ms"]
        r = args["r"]
        xi_hms = args["xihms"]
        icovs = args["icovs"]
        nus = args["nus"] - self.nu_pivot #subtract off the pivot nu
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m

        concs = conc0 - conc1*nus
        alphas = alpha0 + alpha2*nus**2
        rts = rt0 + rt1*nus
        r_As = r_A0 + r_A1*nus
        r_Bs = r_B0 + r_B1*nus
        biases = bias0 + bias1*nus

        #Priors part 2 here
        if any(concs < .1) or any(alphas < .01) or any(rts < 0.1):
            print("in prior 2", concs, alphas, rts)
            return -1e99
        if any(r_As < r_Bs) or any(r_As > 10) or any(r_Bs < 0.0001):
            print("in prior 3", r_As, r_Bs)
            return -1e-99

        print(np.exp(params))
        exit()
        
        #Loop over mass bins
        LL = 0
        for j in range(len(nus)):
            xi_hm = xi_hms[j]
            icov = icovs[j]
            M = Ms[j]
            
            nu = nus[j]
            conc = concs[j]
            alpha = alphas[j]
            rt = rts[j]
            r_A = r_As[j]
            r_B = r_Bs[j]
            bias = biases[j]
            xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha,
                                                      rt, beta,
                                                      r_eff, beta_eff,
                                                      r_A, r_B, beta_ex,
                                                      bias, xi_nl, Omega_m)
            X = xi_hm - xi_ex
            LL += -0.5*np.dot(X, np.dot(icov, X))
            continue
        print LL
        exit()
        return LL

    def perform_best_fit(self, quiet=False, bfoutpath=None):
        if bfoutpath is None:
            bfoutpath = "bestfit_FULLSNAP_Box{0}".format(self.box)
        
        args = self.setup_fitting_args()

        guess = np.array([6.5, #conc0
                                 0.5, #conc1
                                 0.18, #alpha0
                                 0.005, #alpha2
                                 1.8, #rt0
                                 0.1, #rt1
                                 0.25, #beta
                                 1.6, #r_eff
                                 0.7, #beta_eff
                                 1.8, #r_A0
                                 0.1, #r_A1
                                 0.6, #r_B0
                                 0.05, #r_B1
                                 0.28, #beta_ex
                                 2.5, #bias0
                                 0.5, #bias1
        ])
        if self.fit_in_log:
            guess = np.log(guess)
            
        result = op.minimize(lambda *args: -self.lnlike(*args),
                             guess, args=(args,),
                             method=method)

        if not quiet:
            print result

        names = ["c0","c1","a0","a2","rt0","rt1","beta","re","betae","rA0","rA1","rB0","rB1","betax","b0","b1"]
        for i,lnp in enumerate(result.x):
            if self.fit_in_log:
                print("{0}\t {1:.3f}".format(names[i],np.exp(lnp)))
            else:
                print("{0}\t {1:.3f}".format(names[i], lnp))

        if not os.path.isfile(bfoutpath+".npy"):
            Np = len(guess)
            Nz = len(ad.scale_factors())
            bfs = np.zeros((Nz,Np))
        else:
            bfs = np.load(bfoutpath+".npy")
        bfs[self.snapshot] = result.x
        np.save(bfoutpath, bfs)
        return

    def get_bestfit_model(self, mass_index, bfoutpath=None):
        if bfoutpath is None:
            bfoutpath = "bestfit_FULLSNAP_Box{0}.npy".format(self.box)
        snap = self.snapshot
        if self.fit_in_log:
            conc0, conc1, alpha0, alpha2, rt0, rt1,\
                beta, r_eff, beta_eff, r_A0, r_A1, r_B0, r_B1, beta_ex, \
                bias0, bias1 = np.exp(np.load(bfoutpath)[snap])
        else:
            conc0, conc1, alpha0, alpha2, rt0, rt1,\
                beta, r_eff, beta_eff, r_A0, r_A1, r_B0, r_B1, beta_ex, \
                bias0, bias1 = np.load(bfoutpath)[snap]
    
        args = self.setup_fitting_args()
        M = args["Ms"][mass_index]
        r = args["r"]
        xi_hm = args["xihms"][mass_index]
        icov = args["icovs"][mass_index]
        nus = args["nus"] - self.nu_pivot #subtract off the pivot nu
        nu = nus[mass_index]
        xi_nl = args["xi_nl"]
        Omega_m = self.Omega_m

        conc = conc0 - conc1*nu
        alpha = alpha0 + alpha2*nu**2
        rt = rt0 + rt1*nu
        r_A = r_A0 + r_A1*nu
        r_B = r_B0 + r_B1*nu
        bias = bias0 + bias1*nu
        xi_ex = ct.exclusion.xi_hm_exclusion_at_r(r, M, conc, alpha,
                                                  rt, beta,
                                                  r_eff, beta_eff,
                                                  r_A, r_B, beta_ex,
                                                  bias, xi_nl, Omega_m)
        X = xi_hm - xi_ex
        chi2 = np.dot(X, np.dot(icov, X))
        return r, xi_ex, chi2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    box = 0
    zi = 9
    f = SnapshotFitter(box, snapshot=zi, fit_in_log=True)
    f.perform_best_fit()
    
    NM = len(f.data_dict["masses"])
    MI = 2
    for i in range(MI, MI+1):
    #for i in range(NM):
        fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
        r, xi, chi2 = f.get_bestfit_model(i)
        ax[0].loglog(r,xi,zorder=-1,c='b')
        rd = f.data_dict["r"]
        M = f.data_dict["masses"][i]
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
        ax[0].set_title(r"SnapFit box%d Z%d M%d [$M=%.2e$] [$\chi^2=%.2f$]"%(box,zi,i,M,chi2))
        fig.savefig("figs/SNAPFIT_box%d_Z%d_M%d.png"%(box,zi,i))
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()
