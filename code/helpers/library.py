"""
Library function classes and associated functions
"""
import numpy as np

from .utils import reynolds_number


class LibraryFcn2D:
    """2-variable function to populate library for SINDy"""

    def __init__(self, fcn, coeff=0):
        self.fcn = fcn
        self.coeff = coeff

    def __call__(self, h, v):
        return self.fcn(h, v)

    def eval(self, h, v):
        return self.coeff * self.fcn(h, v)


class LibraryFcn3D:
    """3-variable function to populate library for SINDy"""

    def __init__(self, fcn, coeff=0):
        self.fcn = fcn
        self.coeff = coeff

    def __call__(self, t, h, v):
        return self.fcn(t, h, v)

    def eval(self, t, h, v):
        return self.coeff * self.fcn(t, h, v)


def get_library(lib_type="polynomial", order=3):
    """Generate library functions and labels for ball drop data set"""

    if not order == 3:
        raise NotImplementedError(
            "order must be 3 until other options are implemented."
        )
    if lib_type == "polynomial":
        library = {
            "1": LibraryFcn2D(lambda h, v: np.ones_like(h)),
            "x": LibraryFcn2D(lambda h, v: h),
            "v": LibraryFcn2D(lambda h, v: v),
            "xv": LibraryFcn2D(lambda h, v: h * v),
            "x^2": LibraryFcn2D(lambda h, v: h ** 2),
            "v^2": LibraryFcn2D(lambda h, v: v ** 2),
            "x^2v": LibraryFcn2D(lambda h, v: (h ** 2) * v),
            "xv^2": LibraryFcn2D(lambda h, v: h * (v ** 2)),
            "x^3": LibraryFcn2D(lambda h, v: h ** 3),
            "v^3": LibraryFcn2D(lambda h, v: v ** 3),
        }

        return library

    elif lib_type == "polynomial v":
        library = {
            "1": LibraryFcn2D(lambda h, v: np.ones_like(h)),
            "v": LibraryFcn2D(lambda h, v: v),
            "v^2": LibraryFcn2D(lambda h, v: v ** 2),
            "v^3": LibraryFcn2D(lambda h, v: v ** 3),
            "v^4": LibraryFcn2D(lambda h, v: v ** 4),
        }
        return library

    elif lib_type == "nonautonomous":
        library = {
            "1": LibraryFcn3D(lambda t, h, v: np.ones_like(h)),
            "v": LibraryFcn3D(lambda t, h, v: v),
            "v^2": LibraryFcn3D(lambda t, h, v: v ** 2),
            "v^3": LibraryFcn3D(lambda t, h, v: v ** 3),
            "t": LibraryFcn3D(lambda t, h, v: t),
            "tv": LibraryFcn3D(lambda t, h, v: t * v),
            "tv^2": LibraryFcn3D(lambda t, h, v: t * v ** 2),
        }
        return library

    else:
        raise NotImplementedError(
            "lib_type must be polynomial, polynomial v, "
            "or nonautonomous, until other options are implemented"
        )


def get_library_index(library, label):
    """Find index in library of function corresponding to a given label"""
    for idx, name in enumerate(library.keys()):
        if name == label:
            return idx

    return -1


def append_re_dependent_drag(library, diameter, k_viscosity=2 / 3):
    """Append Reynolds number-dependent drag term to the function library."""
    library["v^2Re"] = LibraryFcn2D(
        lambda h, v: (v ** 2) * reynolds_number(v, diameter, k_viscosity)
    )

    return library
