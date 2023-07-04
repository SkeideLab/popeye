"""Try to implement numerosity pRF stimulus, similar to popeye/auditory_stimulus"""

from __future__ import division
import ctypes

import numpy as np

from popeye.base import StimulusModel
from popeye.onetime import auto_attr
import popeye.utilities as utils


def generate_log_numerosity(ts):

    # quantities
    numerosities = np.unique(ts)

    # one-hot vectors
    ohv = np.eye(len(numerosities))

    # translation between numerosity and ohv-> lookup with shape (1,max(num))
    lut = np.zeros((np.max(numerosities)), dtype=int)
    lut[numerosities-1] = np.arange(0, len(numerosities))

    # stimulus timeseries encoded as ohv
    stimulus_ts = ohv[lut[np.array(ts)-1]].T

    return stimulus_ts, np.log(numerosities)


class NumerosityStimulus(StimulusModel):

    def __init__(self, numerosity_ts, tr_length, dtype):
        r"""A child of the StimulusModel class for numerosity stimuli.

        Paramaters
        ----------


        dtype : string
            Sets the data type the stimulus array is cast into.

        tr_length : float
            The repetition time (TR) in seconds.

        """

        StimulusModel.__init__(self, numerosity_ts, dtype, tr_length)

        # absorb the vars
        self.tr_length = tr_length

        stimulus_ts, log_grid = generate_log_numerosity(numerosity_ts)
        # share them
        self.stimuli_oh = utils.generate_shared_array(
            stimulus_ts, ctypes.c_double)
        self.log_grid = utils.generate_shared_array(
            log_grid, ctypes.c_double)
