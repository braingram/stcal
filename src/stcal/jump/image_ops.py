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


def compute_axes(ellipse, expansion, max_radius):
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

    max_radius : float
        Maximum allowable absolute radius.

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
    axis1 = min(axis1, max_radius)
    axis2 = min(axis2, max_radius)

    # FIXME why does this round? it does it now to match the old behavior
    return (round(axis1 / 2) * 2, round(axis2 / 2) * 2)


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
    """
    # For a given DQ plane it will use the list of ellipses to create
    #  expanded ellipses of pixels with
    # the jump flag set.
    _, ngroups, nrows, ncols = gdq_cube.shape
    for ellipse in ellipses:
        # Get the expanded ellipse in a subimage, along with the
        # indices that place this subimage within the full array.
        axis1, axis2 = compute_axes(ellipse, expansion, jump_data.max_extended_radius)

        ys, xs = ellipse_coords(((ellipse[0][0], ellipse[0][1]), (axis1, axis2), ellipse[2]), (nrows, ncols))

        # Propagate forward by num_grps_masked groups.
        for flg_grp in range(grp, min(grp + num_grps_masked + 1, ngroups)):

            # Only propagate the snowball forward to unsaturated pixels.
            sat_mask = (gdq_cube[intg, flg_grp, ys, xs] ^ jump_data.fl_sat).astype(bool)
            gdq_cube[intg, flg_grp, ys[sat_mask], xs[sat_mask]] |= jump_data.fl_jump

    return gdq_cube


def ellipse_coords(ellipse, shape):
    """Compute coordinates for an ellipse

    Parameters
    ----------
    ellipse :

        Ellipse tuple

    shape : (int, int)

        The shape of the full 2D array into which the returned
        subimage should be placed.

    Returns
    -------
    ys : unsigned int array

        Ellipse y coordinates

    xs : unsigned int array

        Ellipse x coordinates
    """

    (yc, xc), (a1, a2), alpha = ellipse
    # to reproduce old behavior
    yc, xc = round(yc), round(xc)
    max_half_axis = max(round(a1 / 2), round(a2 / 2)) + 2

    # Note that the convention between which index is x and which
    # is y is a little confusing here.  To cv2.ellipse, the first
    # coordinate corresponds to the second Python index.  That is
    # why x and y are a bit mixed up below.

    ix1 = max(yc - max_half_axis, 0)
    ix2 = min(yc + max_half_axis + 1, shape[1])
    iy1 = max(xc - max_half_axis, 0)
    iy2 = min(xc + max_half_axis + 1, shape[0])

    image = np.zeros(shape=(iy2 - iy1, ix2 - ix1), dtype=np.uint8)
    image = cv2.ellipse(
        image,
        (yc - ix1, xc - iy1),
        (round(a1 / 2), round(a2 / 2)),
        alpha,
        0,
        360,
        1,
        -1,
    )
    ys, xs = np.where(image != 0)
    return ys + iy1, xs + ix1
