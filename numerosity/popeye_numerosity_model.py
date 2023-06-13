#!/usr/bin/python

""" Classes and functions for fitting numerosity population encoding models """

from __future__ import division
from popeye.spinach import generate_rf_timeseries_1D, generate_og_receptive_field
from popeye.base import PopulationModel, PopulationFit
import popeye.utilities as utils
from popeye.onetime import auto_attr
import nibabel
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import numpy as np
import warnings
warnings.simplefilter("ignore")


class NumerosityModel(PopulationModel):

    def __init__(self, stimulus, hrf_model, normalizer=utils.percent_change, hrf_delay=0):
        r"""A 1D Gaussian population receptive field model [1]_.

        Paramaters
        ----------

        stimulus : `NumerosityStimulus` class object
            A class instantiation of the `NumerosityStimulus` class
            containing a representation of the Numerosity stimulus.

        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`

        """

        # invoke the base class
        PopulationModel.__init__(self, stimulus, hrf_model, normalizer)
        self.hrf_delay = hrf_delay

    def calc_prediction(self, num_pref, tw):
        print(f"Testing numerosity log: {num_pref} and tw: {tw}")
        # receptive field
        # rf = np.exp(-1 * (self.stimulus.log_grid - num_pref) ** 2 /
        #             (2 * tw**2))
        rf = generate_og_receptive_field(
            num_pref, 0, tw, self.stimulus.log_grid[np.newaxis], np.zeros(self.stimulus.log_grid[np.newaxis].shape))

        # evaluate entire RF
        mask = np.ones_like(rf).astype('uint8')

        # extract the response
        response = generate_rf_timeseries_1D(
            self.stimulus.stimuli_oh, rf.squeeze(), mask.squeeze())

        # convolve it with the stimulus
        model = fftconvolve(response, self.hrf())[0:len(response)]

        # units
        model = self.normalizer(model)

        return model

    def generate_ballpark_prediction(self, num_pref, tw):
        r"""
        Generate a prediction for the 1D Gaussian model.

        This function generates a prediction of the 1D Gaussian model,
        given a stimulus and the stimulus-referred model parameters.
    This method does not estimate the scaling
        paramter `beta` or the offset parameter `baseline`, since this
        method will be used for a grid-fit and these values can simply
        be calculated for a particular `pref_num` and `tw` pair.

        Paramaters
        ----------

        pref_num : float
            The preferred numerosity of the 1D Gaussian in log space

        tw : float
            The width of the 1D Gaussian in log space

        """

        model = self.calc_prediction(num_pref, tw)

        # regress out mean and amplitude
        beta, baseline = self.regress(model, self.data)

        # scale
        if not np.isnan(beta):
            model *= beta

        # offset
        if not np.isnan(baseline):
            model += baseline

        return model

    def generate_prediction(self, num_pref, tw, beta, baseline, unscaled=False):
        r"""
        Generate a prediction for the 1D Gaussian auditory pRF model.

        This function generates a prediction of the 1D Gaussian model,
        given a stimulus and the stimulus-referred model parameters.
        This function operates on a spectrogram representation of an
        auditory stimulus.

        Paramaters
        ----------
        """
        model = self.calc_prediction(num_pref, tw)
        if unscaled:
            return model
        else:

            # scale
            if not np.isnan(beta):
                model *= beta

            # offset
            if not np.isnan(baseline):
                model += baseline

            return model


class NumerosityFit(PopulationFit):

    def __init__(self, model, data, grids, bounds,
                 voxel_index=(1, 2, 3), Ns=None, auto_fit=True, verbose=0):
        r"""
        A class containing tools for fitting the 1D Gaussian auditory pRF model.

        The `AuditoryFit` class houses all the fitting tool that are associated with 
        estimatinga pRF model.  The `PopulationFit` takes a `AuditoryModel` instance 
        `model` and a time-series `data`.  In addition, extent and sampling-rate of a 
        brute-force grid-search is set with `grids` and `Ns`.  Use `bounds` to set 
        limits on the search space for each parameter.  

        Paramaters
        ----------


        model : `AuditoryModel` class instance
            An object representing the 1D Gaussian model.

        data : ndarray
            An array containing the measured BOLD signal of a single voxel.

        grids : tuple
            A tuple indicating the search space for the brute-force grid-search.
            The tuple contains pairs of upper and lower bounds for exploring a
            given dimension.  For example `grids=((-10,10),(0,5),)` will
            search the first dimension from -10 to 10 and the second from 0 to 5.
            These values cannot be `None`. 

            For more information, see `scipy.optimize.brute`.

        bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use
            `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would 
            bound the first parameter to be any positive number while the
            second parameter would be bounded between -10 and 10.

        Ns : int
            Number of samples per stimulus dimension to sample during the ballpark search.

            For more information, see `scipy.optimize.brute`.

        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The 
            fitting procedure does not require a voxel index, but 
            collating the results across many voxels will does require voxel
            indices. With voxel indices, the brain volume can be reconstructed 
            using the newly computed model estimates.

        auto_fit : bool
            A flag for automatically running the fitting procedures once the 
            `GaussianFit` object is instantiated.

        verbose : int
            0 = silent
            1 = print the final solution of an error-minimization
            2 = print each error-minimization step

        """

        # invoke the base class
        PopulationFit.__init__(self, model, data, grids, bounds,
                               voxel_index, Ns, auto_fit, verbose)

    @auto_attr
    def overloaded_estimate(self):
        fwhm = self.tw*(2*np.sqrt(2*np.log(2)))
        fwhm = np.exp(self.num_pref+fwhm/2)-np.exp(self.num_pref-fwhm/2)

        return [np.exp(self.num_pref), fwhm, self.tw, self.beta, self.baseline]

    @auto_attr
    def num_pref0(self):
        return self.ballpark[0]

    @auto_attr
    def tw0(self):
        return self.ballpark[1]

    @auto_attr
    def beta0(self):
        return self.ballpark[2]

    @auto_attr
    def baseline0(self):
        return self.ballpark[3]

    @auto_attr
    def num_pref(self):
        return self.estimate[0]

    @auto_attr
    def tw(self):
        return self.estimate[1]

    @auto_attr
    def beta(self):
        return self.estimate[2]

    @auto_attr
    def baseline(self):
        return self.estimate[3]

    @auto_attr
    def receptive_field(self):

        # generate stimulus time-series
        rf = np.exp(-1 * (self.model.stimulus.log_grid - self.num_pref) ** 2 /
                    (2 * self.tw**2))
        return rf
