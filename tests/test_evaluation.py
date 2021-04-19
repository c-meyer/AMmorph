#
# Copyright (c) 2021 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING,
# CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING B. MUENCHEN, GERMANY,
# RIXEN@TUM.DE.
#
# AUTHOR: Christian Meyer, christian.meyer@tum.de
#
# Distributed under 3-Clause BSD license.
# See LICENSE file for more information.
#

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose
from ammorph.evaluation import evaluate


def helper_3d():
    gamma = np.array([1.0, 2.0])
    points = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [2.0, 1.0, 1.0],
                       [0.0, 0.0, 0.0]])
    handles = np.array([[1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]])

    no_of_points = points.shape[0]
    no_of_handles = handles.shape[0]

    out_desired = np.zeros(no_of_points)

    for i in range(no_of_points):
        for j in range(no_of_handles):
            out_desired[i] = out_desired[i] + \
                             gamma[j]*norm(points[i, :] - handles[j, :])**3

    return points, handles, gamma, out_desired


def test_evaluate():
    points, handles, gamma, out_desired = helper_3d()
    phi_func = 'r**3'
    out_actual = evaluate(phi_func, points, handles, gamma)

    assert_allclose(out_actual, out_desired)


def test_evaluate_polynomial_order_0():
    points, handles, gamma, out_desired = helper_3d()
    beta = np.array([0.23])
    out_desired += beta
    phi_func = 'r**3'
    out_actual = evaluate(phi_func, points, handles, gamma, beta, 0)

    assert_allclose(out_actual, out_desired)


def test_evaluate_polynomial_order_1():
    points, handles, gamma, out_desired = helper_3d()
    beta = np.array([0.23, 0.1, 0.2, 0.3])
    out_desired += beta[0]
    for i in range(points.shape[0]):
        out_desired[i] += beta[1:].dot(points[i, :])

    phi_func = 'r**3'
    out_actual = evaluate(phi_func, points, handles, gamma, beta, 1)

    assert_allclose(out_actual, out_desired)

    # Test with out parameter
    out_actual = np.zeros_like(out_desired)
    out_actual_new = evaluate(phi_func, points, handles, gamma, beta, 1,
                              out=out_actual)
    assert id(out_actual) == id(out_actual_new)
    assert_allclose(out_actual, out_desired)
