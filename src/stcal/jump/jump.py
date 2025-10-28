#  jump.py - detect cosmic ray jumps and their side effects like
#            snowballs and showers.

import logging
import multiprocessing
import time

import numpy as np

from .twopoint_difference_class import TwoPointParams
from . import twopoint_difference as twopt
from .snowballs import flag_large_events
from .snowshowers import find_faint_extended

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def detect_jumps_data(jump_data):
    """
    Detect jumps and their side effects, such as showers and snowballs.

    It loads and sets the various input data and parameters needed by each of
    the individual detection methods and then calls the detection methods in
    turn.

    Note that the detection methods are currently set up on the assumption
    that the input science data array will be in units of
    electrons, hence this routine scales those input arrays by the detector
    gain. The methods assume that the read noise values will be in units
    of DN.

    The gain is applied to the science data array using the
    appropriate instrument- and detector-dependent values for each pixel of an
    image.  Also, a 2-dimensional read noise array with appropriate values for
    each pixel is passed to the detection methods.

    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    gdq : int, 4D array
        updated group dq array

    pdq : int, 2D array
        updated pixel dq array

    total_primary_crs : int
        the number of primary cosmic rays found

    number_extended_events : int
        the number of showers or XXX found
    """
    sat, jump, dnu = jump_data.fl_sat, jump_data.fl_jump, jump_data.fl_dnu
    number_extended_events = 0

    pdq = setup_pdq(jump_data)

    # Apply gain to the SCI and readnoise arrays so they're in units
    # of electrons
    data = jump_data.data * jump_data.gain_2d
    gdq = jump_data.gdq
    readnoise_2d = jump_data.rnoise_2d * jump_data.gain_2d

    # also apply to the after_jump thresholds
    # XXX Maybe move this computation
    jump_data.after_jump_flag_e1 = jump_data.after_jump_flag_dn1 * np.nanmedian(jump_data.gain_2d)
    jump_data.after_jump_flag_e2 = jump_data.after_jump_flag_dn2 * np.nanmedian(jump_data.gain_2d)

    # Apply the 2-point difference method as a first pass
    log.info("Executing two-point difference method")
    start = time.time()

    # figure out how many slices to make based on 'max_cores'
    max_available = multiprocessing.cpu_count()
    n_rows = data.shape[2]
    n_slices = calc_num_slices(n_rows, jump_data.max_cores, max_available)

    twopt_params = TwoPointParams(jump_data)
    if n_slices == 1:
        twopt_params.minimum_groups = 3  # XXX Should this be hard coded as 3?
        gdq, row_below_dq, row_above_dq, total_primary_crs = twopt.find_crs(
                    data, gdq, readnoise_2d, twopt_params)
    else:
        gdq, total_primary_crs = twopoint_diff_multi(
            jump_data, twopt_params, data, gdq, readnoise_2d, n_slices)

    # remove redundant bits in pixels that have jump flagged but were
    # already flagged as do_not_use or saturated.
    gdq[gdq & (jump | dnu) == (jump | dnu)] ^= jump
    gdq[gdq & (jump | sat) == (jump | sat)] ^= jump

    #  This is the flag that controls the flagging of snowballs.
    if jump_data.expand_large_events:
        gdq, total_snowballs = flag_large_events(gdq, jump, sat, jump_data)
        log.info("Total snowballs = %i", total_snowballs)
        number_extended_events = total_snowballs  # XXX overwritten

    if jump_data.find_showers:
        gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise_2d, jump_data)
        log.info("Total showers= %i", num_showers)
        number_extended_events = num_showers  # XXX overwritten

    elapsed = time.time() - start
    log.info("Total elapsed time = %g sec", elapsed)

    # Return the updated data quality arrays
    return gdq, pdq, total_primary_crs, number_extended_events


def twopoint_diff_multi(jump_data, twopt_params, data, gdq, readnoise_2d, n_slices):
    """
    Split data for jump detection multiprocessing.
    
    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    twopt_params : TwoPointParams
        Class containing parameters and methods for two point differences.

    data : ndarray
        The science data, 4D array float.

    gdq : ndarray 
        The group DQ, 4D array uint8.

    readnoise_2d : ndarray
        The read noise reference, 2D array float.

    n_slices : int
        The number of data slices for multiprocessing.

    Returns
    -------
    gdq : ndarray
        the group DQ array, 4D uint8

    total_primary_crs : int
        total number of primary cosmic rays computed
    """
    slices, yinc = slice_data(twopt_params, data, gdq, readnoise_2d, n_slices)

    log.info("Creating %d processes for jump detection ", n_slices)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=n_slices)
    ######### JUST FOR DEBUGGING #########################
    # pool = ctx.Pool(processes=1)
    # Starts each slice in its own process. Starmap allows more than one
    # parameter to be passed.
    real_result = pool.starmap(twopt.find_crs, slices)
    pool.close()
    pool.join()

    return reassemble_sliced_data(real_result, jump_data, gdq, yinc)


def reassemble_sliced_data(real_result, jump_data, gdq, yinc):
    """
    Reassemble the data from each process for multiprocessing.

    Parameters
    ----------
    real_result : tuple
        The tuple return values from twopt.find_crs
        (gdq, row_below_gdq, row_above_gdq, num_primary_crs)

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    gdq : ndarray
        The group DQ, 4D array uint8.

    yinc : int
        The number of rows in each slice (rows are the y-axis, so this
        says how many rows to increment to get to the next slice.

    Returns
    -------
    gdq : ndarray
        The group DQ, 4D array uint8.

    total_primary_crs : int
        Total number of primary cosmic rays detected.
    """
    nints, ngroups, nrows, ncols = gdq.shape
    row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    previous_row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    row_below_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)

    # Reconstruct gdq, the row_above_gdq, and the row_below_gdq from the
    # slice result
    total_primary_crs = 0

    # Reassemble the data
    for k, resultslice in enumerate(real_result):
        if len(real_result) == k + 1:  # last result
            gdq[:, :, k * yinc: nrows, :] = resultslice[0]
        else:
            gdq[:, :, k * yinc: (k + 1) * yinc, :] = resultslice[0]
        row_below_gdq[:, :, :] = resultslice[1]
        row_above_gdq[:, :, :] = resultslice[2]
        total_primary_crs += resultslice[3]
        if k != 0:
            # For all but the first slice, flag any CR neighbors in the top
            # row of the previous slice and flag any neighbors in the
            # bottom row of this slice saved from the top of the previous
            # slice
            gdq[:, :, k * yinc - 1, :] |= row_below_gdq[:, :, :]
            gdq[:, :, k * yinc, :] |= previous_row_above_gdq[:, :, :]

        # save the neighbors to be flagged that will be in the next slice
        previous_row_above_gdq = row_above_gdq.copy()

    return gdq, total_primary_crs



def slice_data(twopt_params, data, gdq, readnoise_2d, n_slices):
    """
    Create a slice of data for each process for multiprocessing.

    Parameters
    ----------
    twopt_params : TwoPointParams
        Class containing parameters and methods for two point differences.

    data : ndarray
        The science data, 4D array float.

    gdq : ndarray 
        The group DQ, 4D array uint8.

    readnoise_2d : ndarray
        The read noise reference, 2D array float.

    n_slices : int
        The number of data slices for multiprocessing.

    Returns
    -------
    slices : array
        The array of data slices to be used in multiprocessing

    yinc : int
        The number of rows in each slice (rows are the y-axis, so this
        says how many rows to increment to get to the next slice.
    """
    nrows = data.shape[2]
    yinc = nrows // n_slices
    slices = []
    # Slice up data, gdq, readnoise_2d into slices
    # Each element of slices is a tuple of
    # (data, gdq, readnoise_2d, rejection_thresh, three_grp_thresh,
    #  four_grp_thresh, nframes)
    for i in range(n_slices - 1):
        slices.insert(
            i,
            (
                data[:, :, i * yinc: (i + 1) * yinc, :],
                gdq[:, :, i * yinc: (i + 1) * yinc, :],
                readnoise_2d[i * yinc: (i + 1) * yinc, :],
                twopt_params,
            ),
        )

    # last slice get the rest
    slices.insert(
        n_slices - 1,
        (
            data[:, :, (n_slices - 1) * yinc: nrows, :],
            gdq[:, :, (n_slices - 1) * yinc: nrows, :],
            readnoise_2d[(n_slices - 1) * yinc: nrows, :],
            twopt_params,
        ),
    )
    return slices, yinc


def setup_pdq(jump_data):
    """
    Prepare the pixel DQ array for procesing, removing invalid data.

    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    pdq : ndarray
        The pixel DQ array (2D)
    """
    pdq = jump_data.pdq
    bad_gain = (jump_data.gain_2d <= 0.0) | np.isnan(jump_data.gain_2d)
    pdq[bad_gain] |= (jump_data.fl_ngv | jump_data.fl_dnu)

    return pdq


def calc_num_slices(n_rows, max_cores, max_available):
    """
    Compute the number of data slices needed for multiprocessesing.

    Parameters
    ----------
    n_rows : int
        The number of rows of the science data.

    max_cores : str
        The number of processes requested.

    max_available ; int
        The maximum number of CPU cores available.

    Returns
    -------
    The number of slices to slice the data into.
    """
    n_slices = 1
    if max_cores.isnumeric():
        n_slices = int(max_cores)
    elif max_cores.lower() == "none" or max_cores.lower() == "one":
        n_slices = 1
    elif max_cores == "quarter":
        n_slices = max_available // 4 or 1
    elif max_cores == "half":
        n_slices = max_available // 2 or 1
    elif max_cores == "all":
        n_slices = max_available

    # Make sure we don't have more slices than rows or available cores.
    return min([n_rows, n_slices, max_available])
