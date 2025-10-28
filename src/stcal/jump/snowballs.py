import logging

import numpy as np

from .image_ops import ellipse_coords, extend_ellipses, fit_ellipses

log = logging.getLogger(__name__)


def flag_large_events(gdq, jump_flag, sat_flag, jump_data):
    """
    Control the creation of expanded regions that are flagged as jumps.

    These events are called snowballs for the NIR. While they are most commonly
    circular, there are elliptical ones. This routine does not handle the
    detection of MIRI showers.

    Parameters
    ----------
    gdq : int, 4D array
        Group dq array

    jump_flag : int
        DQ flag for jump detection.

    sat_flag: int
        DQ flag for saturation

    Returns
    -------
    total Snowballs
    """
    log.info("Flagging Snowballs")

    n_showers_grp = []
    total_snowballs = 0
    nints, ngrps, nrows, ncols = gdq.shape
    persist_jumps = np.zeros(shape=(nints, nrows, ncols), dtype=np.uint8)
    for integration in range(nints):
        for group in range(1, ngrps):
            current_gdq = gdq[integration, group, :, :]
            current_sat = np.bitwise_and(current_gdq, sat_flag)

            prev_gdq = gdq[integration, group - 1, :, :]
            prev_sat = np.bitwise_and(prev_gdq, sat_flag)

            not_prev_sat = np.logical_not(prev_sat)
            new_sat = current_sat * not_prev_sat
            if group < ngrps - 1:
                next_gdq = gdq[integration, group + 1, :, :]
                next_sat = np.bitwise_and(next_gdq, sat_flag)
                not_current_sat = np.logical_not(current_sat)
                next_new_sat = next_sat * not_current_sat

            next_sat_ellipses = fit_ellipses(next_new_sat & sat_flag, jump_data.min_sat_area)
            sat_ellipses = fit_ellipses(new_sat & sat_flag, jump_data.min_sat_area)

            # find the ellipse parameters for jump regions
            jump_ellipses = fit_ellipses(
                gdq[integration, group, :, :] & jump_flag, jump_data.min_jump_area)

            if jump_data.sat_required_snowball:
                gdq, snowballs, persist_jumps = make_snowballs(
                    gdq, integration, group, jump_ellipses, sat_ellipses,
                    next_sat_ellipses, jump_data, persist_jumps,
                )
            else:
                snowballs = jump_ellipses
            n_showers_grp.append(len(snowballs))
            total_snowballs += len(snowballs)
            gdq = extend_ellipses(
                gdq, integration, group, snowballs, jump_data,
                jump_data.expand_factor, 0,
            )

    #  Test to see if the flagging of the saturated cores will be
    #  extended into the subsequent integrations. Persist_jumps contains
    #  all the pixels that were saturated in the cores of snowballs.
    if jump_data.mask_persist_grps_next_int:
        for intg in range(1, nints):
            if jump_data.persist_grps_flagged >= 1:
                last_grp_flagged = min(jump_data.persist_grps_flagged, ngrps)
                gdq[intg, 1:last_grp_flagged, :, :] = np.bitwise_or(
                        gdq[intg, 1:last_grp_flagged, :, :],
                        np.repeat(persist_jumps[intg - 1, np.newaxis, :, :],
                        last_grp_flagged - 1, axis=0))
    return gdq, total_snowballs


def make_snowballs(
    gdq, integration, group, jump_ellipses, sat_ellipses,
    next_sat_ellipses, jump_data, persist_jumps
):
    """
    Find snowballs.

    Parameter
    ---------
    gdq : ndarray
        The 4-D group DQ array.

    integration : int
        The current integration being used.

    group : int
        The current group being used.

    jump_ellipses : ellipses
        Ellipses computed based on jump detection.

    sat_ellipses : ellipses
        Ellipses computed based on saturation.

    next_sat_ellipses : ellipses
        Ellipses computed based on saturation in the next group.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    persist_jumps : ndarray
        Zero array to be filled in.

    Returns
    -------
    gdq : ndarray
        The 4-D group DQ array.

    snowballs : list
        List of snowballs found.

    persist_jumps : ndarray
        Filled in array.
    """
    nints, ngroups, nrows, ncols = gdq.shape
    low_threshold = jump_data.edge_size
    high_threshold = max(0, nrows - jump_data.edge_size)

    # This routine will create a list of snowballs (ellipses) that have the
    # center of the saturation circle within the enclosing jump rectangle.
    snowballs = []
    for jump in jump_ellipses:
        if near_edge(jump, low_threshold, high_threshold):
            # if the jump ellipse is near the edge, do not require saturation in the
            # center of the jump ellipse
            snowballs.append(jump)
        else:
            for sat in sat_ellipses:
                if ((point_inside_ellipse(sat[0], jump) and jump not in snowballs)):
                    snowballs.append(jump)
            if group < ngroups - 1:
                # Is there saturation inside the jump in the next group?
                for next_sat in next_sat_ellipses:
                    if ((point_inside_ellipse(next_sat[0], jump)) and jump not in snowballs):
                        snowballs.append(jump)

    # extend the saturated ellipses that are larger than the min_sat_radius
    gdq[integration, :, :, :], persist_jumps[integration, :, :] = extend_saturation(
        gdq[integration, :, :, :],
        group,
        sat_ellipses,
        jump_data,
        persist_jumps[integration, :, :],
    )

    return gdq, snowballs, persist_jumps


def point_inside_ellipse(point, ellipse):
    """
    Detect if a point is inside an ellipse.

    Parameters
    ----------
    point : tuple
        Point of interest.

    ellipse : cv2.ellipse
        Ellipse for testing.

    Returns
    -------
    Boolean decision if point is in ellipse
    """
    delta_center = np.sqrt((point[0] - ellipse[0][0]) ** 2 + (point[1] - ellipse[0][1]) ** 2)
    major_axis = max(ellipse[1][0], ellipse[1][1])

    return delta_center < major_axis


def near_edge(jump, low_threshold, high_threshold):
    """
    Test whether the center of a jump is close to the edge of the detector.

    Jumps that are within the threshold will not require a saturated core
    since this may be off the detector

    Parameters
    ----------
    jump : cv2.ellipse
        Ellipse to check if close to detector edge.

    low_threshold :  int
        Low threshold distance from the edge of the detector where saturated cores are not
        required for snowball detection.

    high_threshold : 
        High threshold distance from the edge of the detector where saturated cores are not
        required for snowball detection.

    Returns
    -------
    Boolean : True if ellipse is close to the detector's edge.
    """
    return (
        jump[0][0] < low_threshold
        or jump[0][1] < low_threshold
        or jump[0][0] > high_threshold
        or jump[0][1] > high_threshold
    )


def extend_saturation(cube, grp, sat_ellipses, jump_data, persist_jumps):
    """
    Extend the saturated ellipses that are larger than the min_sat_radius.

    Parameters
    ----------
    cube : ndarray
        Group DQ cube for an integration.

    grp : int
        The current group.

    sat_ellipses : ellipse
        The saturated ellipse.

    jump_data : JumpData

    persist_jumps : ndarray
        3D (nints, nrows, ncols) uint8

    Returns
    -------
    outcube : ndarray
        Group DQ cube for an integration.

    persist_jumps : ndarray
        3D (nints, nrows, ncols) uint8
    """
    ngroups, nrows, ncols = cube.shape
    satcolor = 22  # (0, 0, 22) is a dark blue in RGB
    for ellipse in sat_ellipses:
        if min(ellipse[1]) > jump_data.min_sat_radius_extend:
            ys, xs = ellipse_coords(
                (
                    (ellipse[0][0], ellipse[0][1]),
                    (
                        min(ellipse[1][0] + jump_data.sat_expand, jump_data.max_extended_radius),
                        min(ellipse[1][1] + jump_data.sat_expand, jump_data.max_extended_radius),
                    ),
                    ellipse[2],
                ),
                (nrows, ncols),
            )

            for i in range(grp, cube.shape[0]):
                cube[i, ys, xs] = jump_data.fl_sat

            # Create another non-extended ellipse that is used to
            # create the persist_jumps for this integration. This
            # will be used to mask groups in subsequent integrations.

            persist_jumps[*ellipse_coords(ellipse, (nrows, ncols))] = jump_data.fl_jump
    return cube, persist_jumps
