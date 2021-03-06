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
from numpy.testing import assert_allclose
from ammorph.assembly import sparse_assemble


def test_sparse_assembly():
    """
    Test assembly with simple rbf function f(r) = r**3
    """
    s = r'r**3'
    # Define two nodes
    x = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    actual = sparse_assemble(s, x)
    desired = np.array([[0.0, np.sqrt(2.0)**3], [np.sqrt(2.0)**3, 0.0]])
    assert_allclose(actual.toarray(), desired)


def test_sparse_assembly_with_polynomial_order_0():
    """
    Test assembly with polynomial order = 0
    """
    s = r'r**3'
    # Define two nodes
    x = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    actual = sparse_assemble(s, x, 0)
    desired = np.array([[0.0, np.sqrt(2.0)**3, 1.0],
                        [np.sqrt(2.0)**3, 0.0, 1.0],
                        [1.0, 1.0, 0.0]])
    assert_allclose(actual.toarray(), desired)


def test_sparse_assembly_with_polynomial_order_0_2d():
    """
    Test assembly with polynomial order = 0
    """
    s = r'r**3'
    # Define two nodes
    x = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    actual = sparse_assemble(s, x, 0)
    desired = np.array([[0.0, np.sqrt(2.0)**3, 1.0],
                        [np.sqrt(2.0)**3, 0.0, 1.0],
                        [1.0, 1.0, 0.0]])
    assert_allclose(actual.toarray(), desired)


def test_sparse_assembly_with_polynomial_order_1():
    """
    Test assembly with polynomial order = 1
    """
    s = r'r**3'
    # Define two nodes
    x = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.2, 0.5, 0.6]])
    actual = sparse_assemble(s, x, 1)
    desired = np.array([[0.0, np.sqrt(5.0)**3,
                         np.sqrt(0.8**2+0.5**2+0.6**2)**3,
                         1.0, 1.0, 0.0, 0.0],
                        [np.sqrt(5.0)**3, 0.0,
                         np.sqrt(0.2**2 + 1.5**2 + 0.6**2)**3,
                         1.0, 0.0, 2.0, 0.0],
                        [np.sqrt(0.8**2+0.5**2+0.6**2)**3,
                         np.sqrt(0.2**2 + 1.5**2 + 0.6**2)**3, 0.0,
                         1.0, 0.2, 0.5, 0.6],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]])
    assert_allclose(actual.toarray(), desired)


def test_sparse_assembly_with_polynomial_order_1_2d():
    """
    Test assembly with polynomial order = 1
    """
    s = r'r**3'
    # Define two nodes
    x = np.array([[1.0, 0.0],
                  [0.0, 2.0],
                  [0.2, 0.5]])
    actual = sparse_assemble(s, x, 1)
    desired = np.array([[0.0, np.sqrt(5.0)**3, np.sqrt(0.8**2+0.5**2)**3,
                         1.0, 1.0, 0.0],
                        [np.sqrt(5.0)**3, 0.0, np.sqrt(0.2**2 + 1.5**2)**3,
                         1.0, 0.0, 2.0],
                        [np.sqrt(0.8**2+0.5**2)**3,
                         np.sqrt(0.2**2 + 1.5**2)**3, 0.0,
                         1.0, 0.2, 0.5],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.5, 0.0, 0.0, 0.0]])
    assert_allclose(actual.toarray(), desired)
