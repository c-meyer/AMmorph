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
from scipy.spatial.distance import cdist
import numexpr as ne


def assemble(rbffunc, points, polynomial_order=None, out=None):
    r"""
    assemble(rbffunc, points, polynomial_order, out)

    assembles the matrix

    .. math::
        M = \begin{bmatrix}
            A & P^T \\
            P & 0
        \end{bmatrix}

    with

    .. math::
        A = \begin{bmatrix}
            \phi(h_1 - h_1) & \phi(h_1 - h_2) & \ldots & \phi(h_1 - h_N) \\
            \phi(h_2 - h_1) & \phi(h_2 - h_2) & \ldots & \phi(h_2 - h_N) \\
            & \vdots & & \\
            \phi(h_N - h_1) & \phi(h_N - h_2) & \ldots & \phi(h_N - h_N) \\
        \end{bmatrix}

    and

    .. math::
        P = \begin{bmatrix} 1 & 1 & \ldots & 1 \\
            x_1 & x_2 & \ldots & x_N \\
            y_1 & y_2 & \ldots & y_N \\
            z_1 & z_2 & \ldots & z_N \\
            x_1^2 & x_2^2 & \ldots & x_N^2
            & \ldots & &
            \end{bmatrix}

    :math:`P` contains monomials up to order of argument function argument
    polynomial_order. If polynomial_order is None, :math:`A` is not augmented
    by any :math:`P`
    matrix. The augmentation with :math:`P` is useful for radial basis
    functions that would lead to a singular matrix :math:`A`.
    The augmentation makes :math:`M` positive definite for these functions.


    Parameters
    ----------
    rbffunc : str
        String that defines the rbf function :math:`\phi(r)` used for assembly.
        The function string must contain the character 'r' as argument
    points : array_like
        point coordinates of the handle points :math:`h_i` for assembly.
        points is an array of dimension :math:`N_H \times d` where :math:`N_H`
        is the number of handle points and :math:`d` is the spatial dimension
    polynomial_order : int, optional
        The interpolation problem is augmented by polynomial terms of order
        polynomial_order. Currently only up to order 1 is supported.
        If None, no polynomial terms are added.
        If zero, only the term '1' is added
    out : array_like, optional
        Array to write the matrix :math:`M` into.

    Returns
    -------
    matrix : array_like
        Assembled matrix
    """
    spatial_dimension = points.shape[1]
    no_of_points = points.shape[0]

    if polynomial_order is None:
        extra_columns = 0
    elif polynomial_order == 0:
        extra_columns = 1
    elif polynomial_order == 1:
        extra_columns = spatial_dimension + 1
    else:
        raise NotImplementedError('Currently only augmentation until'
                                  'polynomial order = 1 is supported')

    matrix_dimension = no_of_points + extra_columns
    if out is None:
        out = np.zeros((matrix_dimension, matrix_dimension), dtype=float)

    if polynomial_order is None:
        cdist(points, points, out=out[:no_of_points, :no_of_points])
    else:
        out[:no_of_points, :no_of_points] = cdist(points, points)
    local_dict = {'r': out}
    ne.evaluate(rbffunc, local_dict)
    out[no_of_points:, no_of_points:] = 0.0
    if polynomial_order is not None:
        if polynomial_order >= 0:
            out[no_of_points, :no_of_points] = 1.0
            out[:no_of_points, no_of_points] = 1.0
        if polynomial_order >= 1:
            out[no_of_points+1:no_of_points+1+spatial_dimension,:no_of_points] =\
                points.T
            out[:no_of_points, no_of_points+1:no_of_points+1+spatial_dimension] =\
                points

    return out
