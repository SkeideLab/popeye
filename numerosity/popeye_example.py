# -------------------------------------------------------------------

from popeye.numerosity.popeye_numerosity_stimulus import NumerosityStimulus
import matplotlib.pyplot as plt
import ctypes
import multiprocessing
import datetime
import numpy as np
import sharedmem
import popeye.og_hrf as og
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
from popeye.numerosity.popeye_numerosity_model import NumerosityModel, NumerosityFit

# seed random number generator so we get the same answers ...
np.random.seed(2764932)
stimulus = NumerosityStimulus(
    np.array([1, 2, 3, 4, 5, 20, 5, 4, 3, 2, 1]), 2.1, float)

model = NumerosityModel(stimulus=stimulus, hrf_model=utils.double_gamma_hrf)
# generate random prf estimate
num_pref = np.log(1)
tw = 0.1
hrf_delay = 0
beta = 1
baseline = 0.5

data = model.generate_prediction(num_pref=num_pref, tw=tw,
                                 beta=beta, baseline=baseline)
# add in some noise
data += np.random.uniform(-data.max()/10, data.max()/10, len(data))

# FIT
# define search grids
# these define min and max of the edge of the initial brute-force search.
x_grid = (-0.5, 5)
s_grid = (0.005, 5.25)

# define search bounds
# these define the boundaries of the final gradient-descent search.
x_bound = (-12.0, 12.0)
s_bound = (0.0025, 12.0)  # smallest sigma is a pixel
b_bound = (1e-8, None)
u_bound = (None, None)

# package the grids and bounds
grids = (x_grid, s_grid)
bounds = (x_bound, s_bound, b_bound, u_bound,)
print('-----> Fitting the model....')

# fit the response
# auto_fit = True fits the model on assignment
# verbose = 0 is silent
# verbose = 1 is a single print
# verbose = 2 is very verbose
fit = NumerosityFit(model, data, grids, bounds, Ns=10,
                    voxel_index=(1, 1, 1), auto_fit=True, verbose=2)

# plot the results
plt.plot(fit.prediction, c='r', lw=3, label='model', zorder=1)
plt.scatter(range(len(fit.data)), fit.data,
            s=30, c='k', label='data', zorder=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.xlim(0, len(fit.data))
plt.legend(loc=0)

# multiprocess 3 voxels
data = [data, data, data]
indices = ([1, 2, 3], [4, 6, 5], [7, 8, 9])
bundle = utils.multiprocess_bundle(NumerosityFit, model, data,
                                   grids, bounds, indices,
                                   auto_fit=True, verbose=1, Ns=3)

# run
print("popeye will analyze %d voxels across %d cores" % (len(bundle), 3))
with sharedmem.Pool(np=3) as pool:
    t1 = datetime.datetime.now()
    output = pool.map(utils.parallel_fit, bundle)
    t2 = datetime.datetime.now()
    delta = t2-t1
    print("popeye multiprocessing finished in %s.%s seconds" %
          (delta.seconds, delta.microseconds))
