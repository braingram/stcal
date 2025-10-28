import logging
import warnings

import numpy as np
from astropy import stats
from astropy.convolution import Ring2DKernel
from scipy import signal

from .image_ops import extend_ellipses, fit_ellipses

log = logging.getLogger(__name__)


def find_faint_extended(
        indata, ingdq, pdq, readnoise_2d, jump_data, min_diffs_for_shower=10):
    """
    Flag groups based on showers detected.

    Parameters
    ----------
      indata : float, 4D array
          Science array.

      gdq : int, 2D array
          Group dq array.

      readnoise_2d : float, 2D array
          Readnoise for all pixels.

    Returns
    -------
    gdq : int, 4D array
        updated group dq array.

    number_ellipse : int
        Total number of showers detected.
    """
    log.info("Flagging Showers")
    refpix_flag = jump_data.fl_ref

    gdq = ingdq.copy()
    data = indata.copy()
    nints, ngrps, nrows, ncols = data.shape

    num_grps_donotuse = count_dnu_groups(gdq, jump_data)

    total_diffs = nints * (ngrps - 1) - num_grps_donotuse
    if total_diffs < min_diffs_for_shower:
        log.warning("Not enough differences for shower detections")
        return ingdq, 0

    data = nan_invalid_data(data, gdq, jump_data)

    refy, refx = np.where(pdq == refpix_flag)
    gdq[:, :, refy, refx] = jump_data.fl_dnu
    first_diffs = np.diff(data, axis=1)
    del data

    all_ellipses = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        read_noise_2 = readnoise_2d**2
        if nints >= jump_data.minimum_sigclip_groups:
            mean, median, stddev = stats.sigma_clipped_stats(first_diffs, sigma=5, axis=0)
        else:
            median_diffs = np.nanmedian(first_diffs, axis=(0, 1))
            sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / jump_data.nframes)

        for intg in range(nints):
            if nints < jump_data.minimum_sigclip_groups:
                # The difference from the median difference for each group
                ratio = diff_meddiff_int(intg, median_diffs, sigma, first_diffs)

            #  The convolution kernel creation
            ring_2D_kernel = Ring2DKernel(
                    jump_data.extend_inner_radius, jump_data.extend_outer_radius)
            first_good_group = find_first_good_group(gdq[intg, :, :, :], jump_data.fl_dnu)
            for grp in range(first_good_group + 1, ngrps):
                if nints >= jump_data.minimum_sigclip_groups:
                    ratio = diff_meddiff_grp(intg, grp, median, stddev, first_diffs)

                ellipses = get_bigellipses(
                        ratio, intg, grp, gdq, pdq, jump_data, ring_2D_kernel)

                if len(ellipses) > 0:
                    # add all the showers for this integration to the list
                    all_ellipses.append([intg, grp, ellipses])

    total_showers = 0

    #  Now we actually do the flagging of the pixels inside showers.
    # This is deferred until all showers are detected. because the showers
    # can flag future groups and would confuse the detection algorithm if
    # we worked on groups that already had some flagged showers.
    for showers in all_ellipses:
        intg, grp, ellipses = showers[:3]
        total_showers += len(ellipses)
        gdq, num = extend_ellipses(
            gdq,
            intg,
            grp,
            ellipses,
            jump_data,
            jump_data.extend_ellipse_expand_ratio,
            jump_data.grps_masked_after_shower,
        )

    gdq = max_flux_showers(jump_data, nints, indata, ingdq, gdq)

    return gdq, total_showers


def count_dnu_groups(gdq, jump_data):
    """
    Count the number of groups are flagged as DO_NOT_USE.

    Parameters
    ----------
    gdq : ndarray
        The group DQ 4D uint8.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    num_grps_donotuse : int
        The number of groups flagged as DO_NOT_USE.
    """
    nints, ngrps = gdq.shape[:2]
    num_grps_donotuse = 0
    for integ in range(nints):
        for grp in range(ngrps):
            if np.all(np.bitwise_and(gdq[integ, grp, :, :], jump_data.fl_dnu)):
                num_grps_donotuse += 1
    return num_grps_donotuse


def nan_invalid_data(data, gdq, jump_data):
    """
    Mark flagged data as invalid by setting the science data to NaN.

    Parameters
    ----------
    data : ndarray
        Science data 4D float

    gdq : ndarray
        Group DQ 4D uint8

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    data : ndarray
        NaN'd cience data 4D float
    """
    jump_dnu_flag = jump_data.fl_jump + jump_data.fl_dnu
    sat_dnu_flag = jump_data.fl_sat + jump_data.fl_dnu
    data[gdq == jump_dnu_flag] = np.nan
    data[gdq == sat_dnu_flag] = np.nan
    data[gdq == jump_data.fl_sat] = np.nan
    data[gdq == jump_data.fl_jump] = np.nan
    data[gdq == jump_data.fl_dnu] = np.nan

    return data


def diff_meddiff_int(intg, median_diffs, sigma, first_diffs_masked):
    """
    Compute the SNR ratio of each difference.

    Parameters
    ----------
    intg : int
        Current intregration

    median_diffs : ndarray
        Median of differences in integration

    sigma : ndarray
        Weighting.

    first_diffs_masked : ndarray
        Masked first differences.

    Returns
    -------
    ratio : ndarray
        SNR ratio
    """

    e_jump = first_diffs_masked[intg] - median_diffs[np.newaxis, :, :]

    # SNR ratio of each diff.
    ratio = np.abs(e_jump) / sigma[np.newaxis, :, :]

    return ratio


def diff_meddiff_grp(intg, grp, median, stddev, first_diffs_masked):
    """
    Find the median difference group.

    Parameters
    ----------
    intg : int
        Current intregration

    grp : int
        Current group

    median : float
        Median computed during sigma clipping.

    stddev : float
        Standard deviation computed during sigma clipping.

    first_diffs_masked : ndarray
        Masked first differences.

    Returns
    -------
    ratio : ndarray
        SNR ratio
    """
    median_diffs = median[grp - 1]
    sigma = stddev[grp - 1]

    # The difference from the median difference for each group
    e_jump = first_diffs_masked[intg] - median_diffs[np.newaxis, :, :]

    # SNR ratio of each diff.
    ratio = np.abs(e_jump) / sigma[np.newaxis, :, :]

    return ratio


def find_first_good_group(int_gdq, do_not_use):
    """
    Find first good group.

    Parameters
    ----------
    int_gdq : ndarray
        Group DQ for an integration 3D uint8.

    do_not_use : int
        The DO_NOT_USE flag.

    Returns
    -------
    first_good_group : ndarray
        The first good group of the pixel integration.
    """
    ngrps = int_gdq.shape[0]
    skip_grp = True
    first_good_group = 0
    for grp in range(ngrps):
        mask = np.bitwise_and(int_gdq[grp], do_not_use)
        skip_grp = np.all(mask)
        if not skip_grp:
            first_good_group = grp
            break

    return first_good_group


def convolve_fast(array, kernel):
    """Convolve an array with a kernel, interpolating over NaNs.
    Faster version of astropy.convolution.convolve(preserve_nan=True)
    Parameters
    ----------
    array : 2D array of floats
        Array for convolution
    kernel : 2D array of floats
        Convolution kernel.  Both dimensions must be odd.
    Returns
    -------
    convolved_array : 2D array of floats
        Convolution of array and kernel, interpolating over NaNs.
    """

    # We will mask nan pixels by setting them to zero.  We
    # will convolve by our kernel, then divide by the weight
    # given by the valid pixels convolved with the kernel in
    # order to normalize.  Finally, we will reset the
    # initially nan pixels to nan.
    #
    # This function is equivalent to
    # convolved_array = astropy.convolution.convolve(array, kernel, preserve_nan=True)
    # but runs in about half the time.

    good = np.isfinite(array)
    array[~good] = 0

    convolved_array = signal.oaconvolve(array, kernel, mode='same')

    # Embed the flag in a larger array to reproduce the behavior at
    # the edge with a fill value of zero.

    padded_good_arr = np.ones((good.shape[0] + kernel.shape[0] - 1,
                               good.shape[1] + kernel.shape[1] - 1))
    n = kernel.shape[0]//2
    padded_good_arr[n:-n, n:-n] = good
    norm = signal.oaconvolve(padded_good_arr, kernel, mode='valid')

    # Avoid dividing by a tiny number due to roundoff error.

    good &= norm > 1e-3*np.mean(kernel)
    convolved_array /= norm

    # Replace NaNs

    convolved_array[~good] = np.nan

    return convolved_array


def get_bigellipses(ratio, intg, grp, gdq, pdq, jump_data, ring_2D_kernel):
    """Perform convolution to find contours larger than a minimum area.

    Parameters
    ----------
    ratio : ndarray

    intg : int
        Current integration

    grp : int
        Current group

    gdq : ndarray
        Group DQ array 4D uint8

    pdq : ndarray
        Pixel DQ array 2D uint32

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    ring_2D_kernel : astropy.convolution.Ring2DKernel
        2D Ring filter kernel

    Returns
    -------
    list
        list of ellipses
    """
    masked_ratio = ratio[grp - 1].copy()
    jump_flag = jump_data.fl_jump
    sat_flag = jump_data.fl_sat
    dnu_flag = jump_data.fl_dnu

    #  mask pixels that are already flagged as jump, sat, or dnu
    combined_pixel_mask = np.bitwise_or(gdq[intg, grp, :, :], pdq[:, :])

    jump_sat_or_dnu = np.bitwise_and(combined_pixel_mask, jump_flag|sat_flag|dnu_flag) != 0
    masked_ratio[jump_sat_or_dnu] = np.nan

    kernel = ring_2D_kernel.array

    masked_smoothed_ratio = convolve_fast(masked_ratio, kernel)

    extended_emission = (masked_smoothed_ratio > jump_data.extend_snr_threshold).astype(np.uint8)

    #  find the contours of the extended emission
    return fit_ellipses(extended_emission, jump_data.extend_min_area)


def max_flux_showers(jump_data, nints, indata, ingdq, gdq):
    """
    Ensure that flagging showers didn't change final fluxes by more than allowed.

    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    nints : int
        The number of integrations

    indata : ndarray
        The input data 4D float.

    ingdq : ndarray
        The input group DQ 4D uint8.

    gdq : ndarray
        The computed group DQ 4D uint8.

    Returns
    -------
    gdq : ndarray
        The computed group DQ 4D uint8.
    """
    # Ensure that flagging showers didn't change final fluxes by more than the allowed amount
    for intg in range(nints):
        # Consider DO_NOT_USE, SATURATION, and JUMP_DET flags
        invalid_flags = jump_data.fl_dnu | jump_data.fl_sat| jump_data.fl_jump

        # Approximate pre-shower rates
        tempdata = indata[intg, :, :, :].copy()
        # Ignore any groups flagged in the original gdq array
        tempdata[ingdq[intg, :, :, :] & invalid_flags != 0] = np.nan
        # Compute group differences
        diff = np.diff(tempdata, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            image1 = np.nanmean(diff, axis=0)
        del tempdata

        # Approximate post-shower rates
        tempdata = indata[intg, :, :, :].copy()
        # Ignore any groups flagged in the shower gdq array
        tempdata[gdq[intg, :, :, :] & invalid_flags != 0] = np.nan
        # Compute group differences
        diff = np.diff(tempdata, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            image2 = np.nanmean(diff, axis=0)
        del tempdata

        # Revert the group flags to the pre-shower flags for any pixels whose rates
        # became NaN or changed by more than the amount reasonable for a real CR shower
        # Note that max_shower_amplitude should now be in DN/group not DN/s
        diff = np.abs(image1 - image2)
        indx = np.where((np.isfinite(diff) == False) | (diff > jump_data.max_shower_amplitude))
        gdq[intg, :, indx[0], indx[1]] = ingdq[intg, :, indx[0], indx[1]]

    return gdq
