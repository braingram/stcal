from libcpp.stack cimport stack

from stcal.ramp_fitting.ols_cas22._core cimport RampFit, RampFits, RampIndex, Thresh
from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef class Pixel:
    cdef Fixed fixed
    cdef float read_noise
    cdef float [:] resultants

    cdef float[:] delta_1, delta_2
    cdef float[:] sigma_1, sigma_2

    cdef float[:] resultants_diff(Pixel self, int offset)
    cdef RampFit fit_ramp(Pixel self, RampIndex ramp)

    cdef float correction(Pixel self, int i, int offset, RampIndex ramp)
    cdef float[:] stats(Pixel self, float slope, RampIndex ramp)
    cdef RampFits fit_ramps(Pixel self, stack[RampIndex] ramps)


cdef Pixel make_pixel(Fixed fixed, float read_noise, float [:] resultants)
