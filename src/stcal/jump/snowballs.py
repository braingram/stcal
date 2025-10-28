import logging

import cv2 as cv
import numpy as np

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

            next_sat_ellipses = find_ellipses(next_new_sat, sat_flag, jump_data.min_sat_area)
            sat_ellipses = find_ellipses(new_sat, sat_flag, jump_data.min_sat_area)

            # find the ellipse parameters for jump regions
            jump_ellipses = find_ellipses(
                gdq[integration, group, :, :], jump_flag, jump_data.min_jump_area)

            if jump_data.sat_required_snowball:
                gdq, snowballs, persist_jumps = make_snowballs(
                    gdq, integration, group, jump_ellipses, sat_ellipses,
                    next_sat_ellipses, jump_data, persist_jumps,
                )
            else:
                snowballs = jump_ellipses
            n_showers_grp.append(len(snowballs))
            total_snowballs += len(snowballs)
            gdq, num_events = extend_ellipses(
                gdq, integration, group, snowballs, jump_data,
                expansion=jump_data.expand_factor, num_grps_masked=0,
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


def find_ellipses(dqplane, bitmask, min_area):
    """
    Find ellipses based on DQ masks in bitmask.

    Parameters
    ----------
    dqplane : ndarray
        2D plane of an integration and group

    bitmask : uint8
        bitmask of DQ flags

    min_area : float
        The minimum area of saturated pixels at the center of a snowball. Only
        contours with area above the minimum will create snowballs.

    Returns 
    -------
    list of computed ellipses
    """
    # Using an input DQ plane this routine will find the groups of pixels with
    # at least the minimum
    # area and return a list of the minimum enclosing ellipse parameters.
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bigcontours = [con for con in contours if cv.contourArea(con) > min_area]

    # minAreaRect is used because fitEllipse requires 5 points and it is
    # possible to have a contour
    # with just 4 points.
    return [cv.minAreaRect(con) for con in bigcontours]


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

    jump_ellipses : cv.ellipses
        Ellipses computed based on jump detection.

    sat_ellipses : cv.ellipses
        Ellipses computed based on saturation.

    next_sat_ellipses : cv.ellipses
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


def extend_ellipses(
    gdq_cube, intg, grp, ellipses, jump_data,
    expansion=1.9, expand_by_ratio=True, num_grps_masked=1,
):
    """
    Extend the ellipses.

    Parameters
    ----------
    gdq_cube : ndarray
        Group DQ cube for an integration.  Modified in-place.

    intg : int
        The current integration.

    grp : int
        The current group.

    ellipses : cv.ellipse
        Ellipses for events.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    expansion : float
        The factor that increases the size of the snowball or enclosed ellipse.

    expand_by_ratio : bool
        Should the ellipse expansion be used?

    num_grps_masked : int
        The number of groups flagged.

    Returns
    -------
    gdq_cube : ndarray
        Computed 3-D group DQ array, modified in-place

    num_ellipses : int
        The number of ellipses passed in as a parameter.
    """
    # For a given DQ plane it will use the list of ellipses to create
    #  expanded ellipses of pixels with
    # the jump flag set.
    _, ngroups, nrows, ncols = gdq_cube.shape
    num_ellipses = len(ellipses)
    for ellipse in ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        axes = compute_axes(expand_by_ratio, ellipse, expansion, jump_data)

        alpha = ellipse[2]

        # Get the expanded ellipse in a subimage, along with the
        # indices that place this subimage within the full array.
        axis1 = axes[0]*2
        axis2 = axes[1]*2
        indx, jump_ellipse = ellipse_subim(
            ceny, cenx, axis1, axis2, alpha, jump_data.fl_jump, (nrows, ncols))
        (iy1, iy2, ix1, ix2) = indx
        
        # Propagate forward by num_grps_masked groups.

        for flg_grp in range(grp, min(grp + num_grps_masked + 1, ngroups)):

            # Only propagate the snowball forward to unsaturated pixels.

            sat_pix = gdq_cube[intg, flg_grp, iy1:iy2, ix1:ix2] & jump_data.fl_sat
            jump_ellipse[sat_pix == jump_data.fl_sat] = 0
            gdq_cube[intg, flg_grp, iy1:iy2, ix1:ix2] |= jump_ellipse

    return gdq_cube, num_ellipses


def ellipse_subim(ceny, cenx, axis1, axis2, alpha, value, shape):
    """Draw a filled ellipse in a small array at a given (returned) location
    Parameters
    ----------
    ceny : float
        Center of the ellipse in y (second axis of an image)
    cenx : float
        Center of the ellipse in x (first axis of an image)
    axis1 : float
        One (full) axis of the ellipse
    axis2 : float
        The other (full) axis of the ellipse
    alpha : float
        Angle (in degrees) between axis1 and x
    value : unsigned 8-bit integer
        Value to fill the image with
    shape : (int, int)
        The shape of the full 2D array into which the returned
        subimage should be placed.
    Returns
    -------
    indx : (int, int, int, int)
        Indices (iy1, iy2, ix1, ix2) such that
        fullimage[iy1:iy2, ix1:ix2] = subimage (see below)
    subimage : 2D 8-bit unsigned int array
        Small image containing the ellipse, goes into fullimage
        as described above.
    """
    yc, xc = round(ceny), round(cenx)

    # How big of a subarray do we need for the subimage?

    dn_over_2 = max(round(axis1/2), round(axis2/2)) + 2

    # Note that the convention between which index is x and which
    # is y is a little confusing here.  To cv.ellipse, the first
    # coordinate corresponds to the second Python index.  That is
    # why x and y are a bit mixed up below.

    ix1 = max(yc - dn_over_2, 0)
    ix2 = min(yc + dn_over_2 + 1, shape[1])
    iy1 = max(xc - dn_over_2, 0)
    iy2 = min(xc + dn_over_2 + 1, shape[0])

    image = np.zeros(shape=(iy2 - iy1, ix2 - ix1, 3), dtype=np.uint8)
    image = cv.ellipse(
        image,
        (yc - ix1, xc - iy1),
        (round(axis1 / 2), round(axis2 / 2)),
        alpha,
        0,
        360,
        (0, 0, value),
        -1,
    )

    # The last ("blue") part contains the filled ellipse that we want.
    subimage = image[:, :, 2]
    return (iy1, iy2, ix1, ix2), subimage


def extend_saturation(cube, grp, sat_ellipses, jump_data, persist_jumps):
    """
    Extend the saturated ellipses that are larger than the min_sat_radius.
    
    Parameters
    ----------
    cube : ndarray
        Group DQ cube for an integration.

    grp : int
        The current group.

    sat_ellipses : cv.ellipse
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
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        cen = (round(ceny), round(cenx))
        minor_axis = min(ellipse[1][1], ellipse[1][0])

        if minor_axis > jump_data.min_sat_radius_extend:
            axis1 = ellipse[1][0] + jump_data.sat_expand
            axis2 = ellipse[1][1] + jump_data.sat_expand
            axis1 = min(axis1, jump_data.max_extended_radius)
            axis2 = min(axis2, jump_data.max_extended_radius)

            alpha = ellipse[2]

            indx, sat_ellipse = ellipse_subim(
                ceny, cenx, axis1, axis2, alpha, satcolor, (nrows, ncols))
            (iy1, iy2, ix1, ix2) = indx

            # Create another non-extended ellipse that is used to
            # create the persist_jumps for this integration. This
            # will be used to mask groups in subsequent integrations.

            is_sat = sat_ellipse == satcolor
            for i in range(grp, cube.shape[0]):
                cube[i][iy1:iy2, ix1:ix2][is_sat] = jump_data.fl_sat

            ax1, ax2 = (ellipse[1][0], ellipse[1][1])
            indx, persist_ellipse = ellipse_subim(
                ceny, cenx, ax1, ax2, alpha, satcolor, (nrows, ncols))
            (iy1, iy2, ix1, ix2) = indx

            persist_mask = persist_ellipse == satcolor
            persist_jumps[iy1:iy2, ix1:ix2][persist_mask] = jump_data.fl_jump

    return cube, persist_jumps


def compute_axes(expand_by_ratio, ellipse, expansion, jump_data):
    """
    Expand the ellipse by the expansion factor.

    The number of pixels added to both axes is the number of pixels added
    to the minor axis. This prevents very large flagged ellipses with high
    axis ratio ellipses. The major and minor axis are not always the same
    index.  Therefore, we have to test to find which is actually the minor axis.

    Parameters
    ----------
    expand_by_ratio : bool
        Should the axes be expanded?

    ellipse : cv2.ellipse
        Ellipse to expand.

    expansion : float
        The factor that increases the size of the snowball or enclosed ellipse.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    axes : tuple
        Expanded and rounded ellipse axes.
    """
    if expand_by_ratio:
        if ellipse[1][1] < ellipse[1][0]:
            axis1 = ellipse[1][0] + (expansion - 1.0) * ellipse[1][1]
            axis2 = ellipse[1][1] * expansion
        else:
            axis1 = ellipse[1][0] * expansion
            axis2 = ellipse[1][1] + (expansion - 1.0) * ellipse[1][0]
    else:
        axis1 = ellipse[1][0] + expansion
        axis2 = ellipse[1][1] + expansion
    axis1 = min(axis1, jump_data.max_extended_radius)
    axis2 = min(axis2, jump_data.max_extended_radius)

    return (round(axis1 / 2), round(axis2 / 2))


