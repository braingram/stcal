import copy
import importlib
from unittest.mock import patch

import gwcs
import numpy as np
import pytest
from astropy.modeling.models import Scale, Shift

from stcal.skymatch import SkyGroup, SkyImage, SkyStats, skymatch
from stcal.skymatch.skymatch import _overlap_matrix

skymatch_module = importlib.import_module("stcal.skymatch.skymatch")

IMAGE_SIZE = (10, 10)
FILL_VALUES = [21, 42]


@pytest.fixture()
def images():
    # a simple 1 arcminute FOV wcs
    scales = [1 / (s * 60) for s in IMAGE_SIZE]
    # shifted 1 degree to avoid dealing with RA wrapping
    wcs = gwcs.WCS(
        [["pixel", (Scale(scales[0]) & Scale(scales[1])) | (Shift(1) & Shift(1))], ["world", None]]
    )
    im = SkyImage(
        np.zeros(IMAGE_SIZE, dtype="f4"),
        np.ones(IMAGE_SIZE, dtype=bool),
        wcs.forward_transform,
        wcs.backward_transform,
        SkyStats("mean"),
    )
    images = [im, SkyGroup([copy.deepcopy(im)])]
    images[0].image[:] = FILL_VALUES[0]
    images[1][0].image[:] = FILL_VALUES[1]
    return images


@pytest.mark.parametrize(
    "method, match_down, skys",
    [
        ("local", True, FILL_VALUES),
        ("local", False, FILL_VALUES),
        ("match", True, [0, 21]),
        ("match", False, [-21, 0]),
        ("global", True, [21, 21]),
        ("global", False, [21, 21]),
        ("global+match", True, FILL_VALUES),
        ("global+match", False, FILL_VALUES),
    ],
)
@pytest.mark.parametrize("subtract", [True, False])
def test_skymatch(images, method, match_down, skys, subtract):
    skymatch(images, method, match_down, subtract)
    assert np.allclose([i.sky for i in images], skys)
    if subtract:
        assert np.allclose(images[0].image, FILL_VALUES[0] - skys[0])
        assert np.allclose(images[1][0].image, FILL_VALUES[1] - skys[1])
    else:
        assert np.allclose(images[0].image, FILL_VALUES[0])
        assert np.allclose(images[1][0].image, FILL_VALUES[1])


@pytest.mark.parametrize(
    "input_images, method, error_class, error_match",
    [
        ([], "foo", ValueError, "Unsupported 'skymethod'"),
        ([1, 2], "global", TypeError, "Each element of the 'images' must be"),
        ([], "global", ValueError, "Argument 'images' must contain at least one image"),
    ],
)
def test_skymatch_errors(input_images, method, error_class, error_match):
    with pytest.raises(error_class, match=error_match):
        skymatch(input_images, method)


@pytest.mark.parametrize("method", ["match", "global+match"])
@pytest.mark.parametrize("match_down", [True, False])
@pytest.mark.parametrize("subtract", [True, False])
def test_skymatch_parallel(images, method, match_down, subtract):
    # Three inputs provide enough pairs to exercise multiple processes.
    images.append(copy.deepcopy(images[0]))
    serial_images = copy.deepcopy(images)
    skymatch(serial_images, method, match_down, subtract)
    with patch.object(skymatch_module, "cpu_count", return_value=2):
        skymatch(images, method, match_down, subtract, max_cores="2")

    for actual, expected in zip(images, serial_images, strict=True):
        assert actual.sky == pytest.approx(expected.sky)
        assert actual.is_sky_valid == expected.is_sky_valid
        if isinstance(actual, SkyGroup):
            actual, expected = actual[0], expected[0]
        np.testing.assert_allclose(actual.image, expected.image)


@pytest.mark.parametrize("apply_sky", [True, False])
@pytest.mark.parametrize("invalid_overlap", [True, False])
def test_overlap_matrix_parallel(images, apply_sky, invalid_overlap):
    images.append(copy.deepcopy(images[0]))
    images[0].sky = 5
    images[1].sky = 10
    images[2].sky = 15
    if invalid_overlap:
        images[1][0].mask[:] = False

    expected = _overlap_matrix(images, apply_sky=apply_sky)
    with patch.object(skymatch_module, "cpu_count", return_value=2):
        actual = _overlap_matrix(images, apply_sky=apply_sky, max_cores="2")
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(np.diag(result), 0)
        if invalid_overlap:
            np.testing.assert_array_equal(result[1, :], 0)
            np.testing.assert_array_equal(result[:, 1], 0)
    assert actual[0][2, 0] == FILL_VALUES[0] - (5 if apply_sky else 0)
    assert actual[0][0, 2] == FILL_VALUES[0] - (15 if apply_sky else 0)


@pytest.mark.parametrize("max_cores", ["none", "one", "1", "quarter", "half", "all", "8"])
def test_overlap_matrix_single_pair(images, max_cores):
    with (
        patch.object(skymatch_module, "cpu_count", return_value=8),
        patch.object(skymatch_module.multiprocessing, "get_context") as get_context,
    ):
        _overlap_matrix(images, max_cores=max_cores)
    get_context.assert_not_called()


@pytest.mark.parametrize("max_cores", ["none", "all"])
@pytest.mark.parametrize("nimages", [0, 1])
def test_overlap_matrix_no_pairs(images, max_cores, nimages):
    with patch.object(skymatch_module.multiprocessing, "get_context") as get_context:
        matrices = _overlap_matrix(images[:nimages], max_cores=max_cores)
    get_context.assert_not_called()
    for matrix in matrices:
        np.testing.assert_array_equal(matrix, np.zeros((nimages, nimages)))


def test_overlap_matrix_default_no_multiprocessing(images):
    images.append(copy.deepcopy(images[0]))
    with patch.object(skymatch_module.multiprocessing, "get_context") as get_context:
        _overlap_matrix(images)
    get_context.assert_not_called()
