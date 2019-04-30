import numpy as np
import matplotlib.pyplot as plt

box = 0
snapshot = 9

names = ["c", "alpha", "rt", "beta", "r_eff", "beta_eff", "r_A", "r_B", "beta_ex", "bias"]

masses = np.load("../datafiles/masses_Box%d_Z%d.npy"%(box, snapshot))
bfoutpath = "bestfit_values_Box%d_Z%d.npy"%(box, snapshot)
bfs = np.load(bfoutpath)

print bfs.shape, masses.shape

for i in range(10):
    plt.plot(masses, bfs[:,i], label=names[i])
plt.xscale("log")
plt.legend()
plt.ylabel(r"Parameter")
plt.xlabel(r"Mass [$h^{-1}{\rm M}_\odot$]")
plt.savefig("parameter_change.png", dpi=300, bbox_inches="tight")
plt.show()
