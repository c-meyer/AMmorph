import numpy as np
import sympy
from numpy.testing import assert_allclose

from ammorph import Study
from ammorph.handler import RadialStretchDeformationHandler


def test_radial_stretch_deformation_handler():
    study = Study()
    handle_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    x0 = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                  [-1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                   [0.0, 0.0, 1.0]])
    d0 = np.zeros_like(x0)
    delta_d = np.zeros_like(x0)
    # Get initial coordinate system
    csys_ref = study.csys
    csys_new = csys_ref
    # Create deformation handler
    dh = RadialStretchDeformationHandler(handle_nodes, csys_ref, csys_new)
    # Apply Stretch of amplitude 2.0
    dh.deform(np.array([2.0]), x0, d0, delta_d)
    delta_d_desired = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                                [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                                [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
                                [0.0, 0.0, 0.0]])
    assert_allclose(delta_d, delta_d_desired, rtol=1e-8)


def test_radial_stretch_deformation_handler_rotated_csys():
    study = Study()
    handle_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    x0 = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                  [-1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                   [0.0, 0.0, 1.0]])
    d0 = np.zeros_like(x0)
    delta_d = np.zeros_like(x0)
    # Get initial coordinate system
    csys_ref = study.csys
    csys_new = csys_ref.orient_new_axis('B', sympy.pi/2, csys_ref.j)
    # Create deformation handler
    dh = RadialStretchDeformationHandler(handle_nodes, csys_ref, csys_new)
    # Apply Stretch of amplitude 2.0
    dh.deform(np.array([2.0]), x0, d0, delta_d)
    delta_d_desired = np.array([[0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                                [0.0, 1.0/np.sqrt(2), 0.0],
                                [0.0, 1.0/np.sqrt(2), 0.0],
                                [0.0, 0.0, 1.0]])
    assert_allclose(delta_d, delta_d_desired, rtol=1e-8)


def test_radial_stretch_deformation_handler_shifted_csys():
    study = Study()
    handle_nodes = np.array([0, 1, 4], dtype=np.int32)
    x0 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                   [-1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                   [0.0, 0.0, 1.0]])
    d0 = np.zeros_like(x0)
    delta_d = np.zeros_like(x0)
    # Get initial coordinate system
    csys_ref = study.csys
    csys_new = csys_ref.locate_new('B', 1.0*csys_ref.i)
    # Create deformation handler
    dh = RadialStretchDeformationHandler(handle_nodes, csys_ref, csys_new)
    # Apply Stretch of amplitude 2.0
    dh.deform(np.array([2.0]), x0, d0, delta_d)
    delta_d_desired = np.array([[0.0, 0.0, 0.0],
                               [-1.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0]])
    assert_allclose(delta_d, delta_d_desired, rtol=1e-8)
