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


def evaluate(rbffunc, points, centers, gamma, beta=None, polynomial_order=None,
             out=None):
    r"""evaluate the RBF-interpolation at points with centers and weights

    Evaluates the interpolation according to formula

    .. math::
        s_i = \sum_j^{N_H} \gamma_j \cdot \phi(\| \boldsymbol{x}_i -
        \boldsymbol{h}_j \|) + \beta_0 + \beta_1 x_i + \beta_2 y_i +
        \beta_3 z_i + \beta_4 x_i^2 + \ldots

    Parameters
    ----------
    rbffunc : str
        String that defines the rbf function :math:`\phi(r)` used for assembly.
        The function string must contain the character 'r' as argument.
    points : array_like
        point coordinates of the active points :math:`x_i` to interpolate.
        points is an array of dimension :math:`N_A \times d` where :math:`N_A`
        is the number of active points and :math:`d` is the spatial dimension.
    centers : array_like
        point coordinates of the handle points :math:`h_i` used as centers for
        the radial basis functions.
        centers is an array of dimension :math:`N_H \times d` where :math:`N_H`
        is the number of handle points and :math:`d` is the spatial dimension.
    gamma : array_like
        weights to weight the radial basis function for interpolation.
        gamma is an array of dimension :math:'N_H'.
    beta : array_like, optional
        If polynomial order is not None, beta must be specified.
        It is an array for weighting the augmentation terms (monomials)
    polynomial_order : int, optional
        If None (default), no polynomial terms are used as augmentation
        for interpolation. If int, polynomial terms up to order
        polynomial_order are added as augmentation. In this case beta must be
        specified for defining the weights.
    out : array_like, optional
        Array to write the interpolated redult into (optional)

    Returns
    -------
    s : array_like
        Interpolated values dimension :math:`N_A`
    """
    spatial_dimension = points.shape[1]
    no_of_points = points.shape[0]
    no_of_centers = centers.shape[0]

    if out is None:
        out = np.zeros(no_of_points, dtype=float)

    distances = np.zeros((no_of_points, no_of_centers), dtype=float)
    cdist(points, centers, out=distances)
    local_dict = {'r': distances}
    ne.evaluate(rbffunc, local_dict, out=distances)
    distances.dot(gamma, out=out)

    if polynomial_order is not None:
        if polynomial_order >= 0:
            out += beta[0]
        if polynomial_order >= 1:
            out += points @ beta[1:]
    return out
