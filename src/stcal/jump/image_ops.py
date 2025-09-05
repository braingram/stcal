"""
Zooming out a bit more (from the findContours notes below).
The new find_areas is called by:
    - find_ellipses
    - get_bigellipses
Each are in different contexts.

find_ellipses
=============
Only used in flag_large_events,
which is only used by detect_jumps_data which IS used by jwst (I think this is the only public API)

get_bigellipses
===============
Only used in find_faint_extended,
which is only used by detect_jumps_data (see above, IS used by jwst)

Each of the above returns:
    - gdq (modified in place in function, no need to return)
    - total_snowballs or num_showers, each is logged and both
      overwrite number_extended_events (which is returned)

flag_large_events
=================
iterates through integrations and groups
computes image, uses provided sat_flag(dq flag), uses provided area thresholds
image is either new_sat, next_new_sat, gdq (which uses jump_flag instead of sat_flag)
... etc

find_faint_extended
===================
"""
import numpy as np

import cv2 as cv

import skimage.measure
from skimage.measure import find_contours


def cv_ellipse(image, center, axes, angle, color):
    # startAngle = always 0
    # endAngle = always 360
    # thickness = always -1
    # color (bgr) = (0, 0, value) # only value changes
    return cv.ellipse(image, center, axes, angle, 0, 360, color, -1)


def findContours(image):
    # mode always RETR_EXTERNAL
    # method always CHAIN_APPROX_SIMPLE
    # used in 2 places (neither used externally):
    # - find_ellipses
    # - get_bigcontours
    # in both cases there is:
    # - contour finding
    # - contour filtering based on contourArea
    # then for find_ellipses the contours are passed through minAreaRect before return
    # for get_bigcontours the contours themselves are returned
    # get_bigcontours is used in 1 place (in find_faint_extended) which then
    # immediately calls minAreaRect on these
    #
    # so in all cases it's:
    # - findContours
    # - filter by area (using contourArea)
    # - minAreaRect
    #
    # so the only thing needed is the minAreaRect result
    return cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


def contourArea(contour):
    return cv.contourArea(contour)


def minAreaRect(contour):
    return cv.minAreaRect(contour)


def cv_find_areas(image, threshold):
    cv_contours, _ = findContours(image)
    cv_big_contours = [con for con in cv_contours if contourArea(con) > threshold]
    cv_min_areas = [minAreaRect(con) for con in cv_big_contours]
    return cv_min_areas


def sk_ellipse(image, center, axes, angle, color):
    # The sub-pixel fudge here is to nudge
    # the skimage ellipse slightly closer to
    # what opencv would produce. There is no
    # exact match (as far as I can tell) but
    # empirically this made things "close"
    # where the skimage ellipse had some extra
    # and some missing edge pixels (rather
    # than only missing edge pixels without the fudge).
    rr, cc = skimage.draw.ellipse(
        center[1], center[0],
        axes[1] + 0.25, axes[0] + 0.25,
        image.shape,
        -np.radians(angle),
    )
    image[rr, cc] = color
    return image


def sk_find_area(image, threshold):
    lim = skimage.measure.label(image)
    min_areas = []
    for region in skimage.measure.regionprops(lim):
        w = region.axis_major_length - 1
        h = region.axis_minor_length - 1
        # region.area returns the number of pixels in the region
        # this does not match cv.contourArea which instead returns
        # a smaller value (proportionately much smaller for small contours
        # where a 2x2 pixel region returns a contourArea of 1).
        # So instead we compute a different area here
        #if (w * h) * region.solidity < threshold:
        if (w * h) < threshold:
            continue
        # opencv returns
        # [[cy, cx], [dy, dx], [angle]]
        # where angle is in degrees, not sure what 0 is
        min_areas.append([
            [float(region.centroid[1]), float(region.centroid[0])],
            [h, w],
            -np.degrees(region.orientation)])
    return min_areas


def ellipse(image, center, axes, angle, color):
    #return cv_ellipse(image, center, axes, angle, color)
    return sk_ellipse(image, center, axes, angle, color)


def find_areas(image, threshold):
    #return cv_find_areas(image, threshold)
    return sk_find_area(image, threshold)
