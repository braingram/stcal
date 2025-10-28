import numpy as np

import cv2


def fit_ellipses(pixels, min_area):
    """
    Find ellipses in an image

    Parameters
    ----------
    pixels : ndarray
        2D image

    min_area : float
        The minimum area of fitted ellipses.

    Returns
    -------
    list of computed ellipses
    """

    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.minAreaRect(con) for con in contours if cv2.contourArea(con) > min_area]


def compute_axes(ellipse, expansion, jump_data):
    """
    Expand the ellipse by the expansion factor.

    The number of pixels added to both axes is the number of pixels added
    to the minor axis. This prevents very large flagged ellipses with high
    axis ratio ellipses. The major and minor axis are not always the same
    index.  Therefore, we have to test to find which is actually the minor axis.

    Parameters
    ----------
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
    if ellipse[1][1] < ellipse[1][0]:
        axis1 = ellipse[1][0] + (expansion - 1.0) * ellipse[1][1]
        axis2 = ellipse[1][1] * expansion
    else:
        axis1 = ellipse[1][0] * expansion
        axis2 = ellipse[1][1] + (expansion - 1.0) * ellipse[1][0]
    axis1 = min(axis1, jump_data.max_extended_radius)
    axis2 = min(axis2, jump_data.max_extended_radius)

    return (round(axis1 / 2), round(axis2 / 2))


def extend_ellipses(
    gdq_cube, intg, grp, ellipses, jump_data,
    expansion, num_grps_masked,
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

    ellipses : ellipse
        Ellipses for events.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    expansion : float
        The factor that increases the size of the snowball or enclosed ellipse.

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
        axes = compute_axes(ellipse, expansion, jump_data)

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
    # is y is a little confusing here.  To cv2.ellipse, the first
    # coordinate corresponds to the second Python index.  That is
    # why x and y are a bit mixed up below.

    ix1 = max(yc - dn_over_2, 0)
    ix2 = min(yc + dn_over_2 + 1, shape[1])
    iy1 = max(xc - dn_over_2, 0)
    iy2 = min(xc + dn_over_2 + 1, shape[0])

    image = np.zeros(shape=(iy2 - iy1, ix2 - ix1, 3), dtype=np.uint8)
    image = cv2.ellipse(
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
