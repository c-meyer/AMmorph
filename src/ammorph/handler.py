#
# Copyright (c) 2021 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
# AUTHOR: Christian Meyer
#
from abc import ABC, abstractmethod
import numpy as np
import sympy
from sympy.vector import CoordSys3D

__all__ = ['ShiftDeformationHandler',
           'RotationDeformationHandler',
           'VectorDeformationHandler',
           'StretchDeformationHandler',
           'RadialStretchDeformationHandler'
           ]

class DeformationHandler(ABC):
    """
    Abstract base class for definition deformation of handle nodes.

    Subclasses must implement
    1. a constructor containing handle_nodes
       as numpy array (dtype=int32) as argument and calling the superclass
       super().__init__(handle_nodes)
    2. the deform method with signature
       deform(self, parameters, x0, d0, delta_d)
       where parameters is an array of floats that can be used as parameters
       for the deformation. These paramters will be passed by the stage.
       x0 is the array of reference coordinates that can be used for
       parameterization. d0 is the displacement of the nodes at the beginning
       of the stage. delta_d is an array to write the displacements of the
       handle nodes into.

    Example
    -------
    Write a deformation handler that adds a parametric displacement in z
    direction::

        class ShiftZDeformationHandler(DeformationHandler):
            def __init__(handle_nodes):
                super().__init__(handle_nodes)

            def deform(self, parameters, x0, d0, delta_d):
                delta_d[self._handle_nodes, 2] = delta_d[self._handle_nodes, 2] + parameters[0]

    """
    def __init__(self, handle_nodes):
        self._handle_nodes = handle_nodes

    @property
    def handle_nodes(self):
        """
        Property that returns the indices of the handle nodes i.e. those nodes
        where the deformation is applied.

        Returns
        -------
        handle_nodes : array_like
            Returns the indices of the handle_nodes that are set by this
            deformation handler.
        """
        return self._handle_nodes

    @abstractmethod
    def deform(self, parameters, x0, d0, delta_d):
        """
        Method that applies the desired deformation to delta_d

        Parameters
        ----------
        parameters : array_like
            Array of dtype float that is passed to parameterize the
            deformation.
        x0 : array_like
            Node coordinates in reference configuration.
        d0 : array_like
            Displacement of the node coordinates at the beginning of the stage.
        delta_d : array_like
            Array that is used to write the prescribed deformation into.
            This array is changed at indices handle_nodes.
            The deformation is added to the array. Thus you need to zero
            this array before calling deform if you only want the effect
            of this deformation only.

        Returns
        -------

        """
        raise NotImplementedError('You must implement this method in '
                                  'subclasses.')


class ShiftDeformationHandler(DeformationHandler):
    """
    ShiftDeformationHandler.

    Adds a parametric shift into z-direction of a given sympy coordinate
    system.

    Examples
    --------

    >>> handle_nodes = np.array([0, 1, 2], dtype=np.int32)
    >>> x0 = np.random.rand(9).reshape(3,3)
    >>> d0 = np.zeros_like(x0)
    >>> delta_d = np.zeros_like(x0)
    >>> # Get initial coordinate system
    >>> csys_ref = study.csys
    >>> # Rotate 90 degree around initial x axis:
    >>> csys_new = csys_ref.orient_new_axis('B', sympy.pi/2.0, csys_ref.i)
    >>> # Create deformation handler
    >>> dh = ShiftDeformationHandler(handle_nodes, csys_ref, csys_new)
    >>> # Apply shift of amplitude 2.0
    >>> dh.deform(np.array([2.0]), x0, d0, delta_d)

    """
    def __init__(self, handle_nodes, csys_ref, csys):
        """
        Parameters
        ----------
        handle_nodes : array_like
            array of dtype int32 containing the indices of the handle nodes
            that are to be set by the deformation handler
        csys_ref : CoordSys3D
            sympy coordinate system describing a reference system.
            Usually you pass study.csys as this argument.
        csys : CoordSys3D
            sympy coordinate system that is related to csys_ref.
            The z-axis of this system should point into the direction of the
            desired shift deformation.
        """
        super().__init__(handle_nodes)
        self._csys0 = csys_ref
        self._csys = csys

    def deform(self, parameters, x0, d0, delta_d):
        """
        Writes displacements into delta_d describing the deformation.

        Parameters
        ----------
        parameters : array_like
            Array dtype float containing the parameter values of the desired
            deformation. This handler expects just one parameter at
            parameters[0]. This parameter is the amplitude of the displacement.
        x0 : array_like
            Array dtype float containing the node coordinates of all nodes
            in reference configuration.
        d0 : array_like
            Array dtype float containing the nodal displacement at the
            beginning of the deformation
        delta_d : array_like
            Array to write the result into. The result is added to
            delta_d. Thus, you need to zero this array before you call this
            method, if you want the effect of the DeformationHandler
            only.

        Returns
        -------

        """
        shift_vec = parameters[0]*self._csys.k
        shift_point = self._csys0.origin.locate_new('s', shift_vec)
        shift_coords = shift_point.express_coordinates(self._csys0)
        delta_d[self._handle_nodes, 0] = delta_d[self._handle_nodes, 0] + shift_coords[0]
        delta_d[self._handle_nodes, 1] = delta_d[self._handle_nodes, 1] + shift_coords[1]
        delta_d[self._handle_nodes, 2] = delta_d[self._handle_nodes, 2] + shift_coords[2]


class RotationDeformationHandler(DeformationHandler):
    """
    RotationDeformationHandler applies a rigid body rotation to the deformed
    configuration.

    The rotation axis is the z axis of the coordinate system passed to the
    constructor.
    """
    def __init__(self, handle_nodes, csys_ref, csys):
        """
        Parameters
        ----------
        handle_nodes : array_like
            array of dtype int32 containing the indices of the handle nodes
            that are to be set by the deformation handler
        csys_ref : CoordSys3D
            sympy coordinate system describing a reference system.
            Usually you pass study.csys as this argument.
        csys : CoordSys3D
            sympy coordinate system that is related to csys_ref.
            The z-axis of this system should point into the direction of the
            desired rotation axis.
        """
        super().__init__(handle_nodes)
        self._csys0 = csys_ref
        self._csys = csys

    def deform(self, parameters, x0, d0, delta_d):
        """
        Writes displacements into delta_d describing the deformation.

        Parameters
        ----------
        parameters : array_like
            Array dtype float containing the parameter values of the desired
            deformation. This handler expects just one parameter at
            parameters[0]. This parameter is the angle of rotation around
            the z-axis in radiant.
        x0 : array_like
            Array dtype float containing the node coordinates of all nodes
            in reference configuration.
        d0 : array_like
            Array dtype float containing the nodal displacement at the
            beginning of the deformation
        delta_d : array_like
            Array to write the result into. The result is added to
            delta_d. Thus, you need to zero this array before you call this
            method, if you want the effect of the DeformationHandler
            only.

        Returns
        -------

        """
        x = x0 + d0
        # define rotated coordinate sytem
        __csys_r = self._csys.orient_new_axis('__csys_r', parameters[0], self._csys.k)
        # define symbols
        __rdh_x = sympy.symbols('__rdh_x')
        __rdh_y = sympy.symbols('__rdh_y')
        __rdh_z = sympy.symbols('__rdh_z')
        __rdh_point = self._csys0.origin.locate_new('__rdh_point', __rdh_x * self._csys0.i + __rdh_y * self._csys0.j + __rdh_z * self._csys0.k)
        __rdh_xi = __rdh_point.express_coordinates(self._csys)
        __rdh_xi_new = self._csys.origin.locate_new('__rdh_xi_new', __rdh_xi[0] * __csys_r.i + __rdh_xi[1] * __csys_r.j + __rdh_xi[2] * __csys_r.k)
        __x_new = __rdh_xi_new.express_coordinates(self._csys0)

        fun = sympy.lambdify((__rdh_x, __rdh_y, __rdh_z), __x_new)
        new_x = np.stack(fun(x[self._handle_nodes,0], x[self._handle_nodes,1], x[self._handle_nodes,2])).T
        delta_d[self._handle_nodes, :] = delta_d[self._handle_nodes, :] + new_x - x[self._handle_nodes, :]


class VectorDeformationHandler(DeformationHandler):
    """
    VectorDeformationHandler.

    Adds a deformation where the displacement of each coordinate is prescribed
    by the coordinates of a vector.

    Examples
    --------

    >>> handle_nodes = np.array([0, 1, 2], dtype=np.int32)
    >>> deformation = np.random.rand(9).reshape(3,3)*0.1
    >>> x0 = np.random.rand(9).reshape(3,3)
    >>> d0 = np.zeros_like(x0)
    >>> delta_d = np.zeros_like(x0)
    >>> # Create deformation handler
    >>> dh = VectorDeformationHandler(handle_nodes, deformation)
    >>> dh.deform(np.array([]), x0, d0, delta_d)

    """
    def __init__(self, handle_nodes, deformation):
        """
        Parameters
        ----------
        handle_nodes : array_like
            array of dtype int32 containing the indices of the handle nodes
            that are to be set by the deformation handler
        deformation : array_like
            array of dtype float64 dimension (handle_nodes, 3) containing
            the displacement of the prescribed deformation
        """
        super().__init__(handle_nodes)
        self._deformation = deformation

    def deform(self, parameters, x0, d0, delta_d):
        """
        Writes displacements into delta_d describing the deformation.

        Parameters
        ----------
        parameters : array_like
            Array dtype float containing the parameter values of the desired
            deformation. This handler expects just one parameter at
            parameters[0]. This parameter is the amplitude of the displacement.
        x0 : array_like
            Array dtype float containing the node coordinates of all nodes
            in reference configuration.
        d0 : array_like
            Array dtype float containing the nodal displacement at the
            beginning of the deformation
        delta_d : array_like
            Array to write the result into. The result is added to
            delta_d. Thus, you need to zero this array before you call this
            method, if you want the effect of the DeformationHandler
            only.

        Returns
        -------

        """
        delta_d[self._handle_nodes, 0] = delta_d[self._handle_nodes, 0] + self._deformation[:, 0]
        delta_d[self._handle_nodes, 1] = delta_d[self._handle_nodes, 1] + self._deformation[:, 1]
        delta_d[self._handle_nodes, 2] = delta_d[self._handle_nodes, 2] + self._deformation[:, 2]


class StretchDeformationHandler(DeformationHandler):
    """
    StretchDeformationHandler.

    Stretches elements linearly to their distance in z direction to a reference sympy coordinate system.

    Examples
    --------

    >>> handle_nodes = np.array([0, 1, 2], dtype=np.int32)
    >>> x0 = np.random.rand(9).reshape(3,3)
    >>> d0 = np.zeros_like(x0)
    >>> delta_d = np.zeros_like(x0)
    >>> # Get initial coordinate system
    >>> csys_ref = study.csys
    >>> # Rotate 90 degree around initial x axis:
    >>> csys_new = csys_ref.orient_new_axis('B', sympy.pi/2.0, csys_ref.i)
    >>> # Create deformation handler
    >>> dh = StretchDeformationHandler(handle_nodes, csys_ref, csys_new)
    >>> # Apply a stretch of amplitude 2.0
    >>> dh.deform(np.array([2.0]), x0, d0, delta_d)

    """
    def __init__(self, handle_nodes, csys_ref, csys, mode='factor'):
        """
        Parameters
        ----------
        handle_nodes : array_like
            array of dtype int32 containing the indices of the handle nodes
            that are to be set by the deformation handler
        csys_ref : CoordSys3D
            sympy coordinate system describing a reference system.
            Usually you pass study.csys as this argument.
        csys : CoordSys3D
            sympy coordinate system that is related to csys_ref.
            The z-axis of this system should point into the direction of the
            desired shift deformation.
        mode: {'factor', 'maximum'}
            if mode = factor (default): the parameter is a stretch factor.
            if mode = maximum: the parameter is a maximum delta for the node with the highest distance to csys.
        """
        super().__init__(handle_nodes)
        if mode in ['stretch', 'max_delta']:
            self._mode = mode
        else:
            raise ValueError('mode must be either "factor" or "max_delta"')
        self._csys0 = csys_ref
        self._csys = csys

    def deform(self, parameters, x0, d0, delta_d):
        """
        Writes displacements into delta_d describing the deformation.

        Parameters
        ----------
        parameters : array_like
            Array dtype float containing the parameter values of the desired
            deformation. This handler expects just one parameter at
            parameters[0]. This parameter is the amplitude of the displacement.
        x0 : array_like
            Array dtype float containing the node coordinates of all nodes
            in reference configuration.
        d0 : array_like
            Array dtype float containing the nodal displacement at the
            beginning of the deformation
        delta_d : array_like
            Array to write the result into. The result is added to
            delta_d. Thus, you need to zero this array before you call this
            method, if you want the effect of the DeformationHandler
            only.

        Returns
        -------

        """
        # 1. Compute the distance to reference:
        coords = np.array(self._csys.origin.express_coordinates(self._csys0)).astype(float)
        delta_x = x0 - coords
        zvec = self._csys0.origin.locate_new('zvec', 1 * self._csys.k)
        zvec_0 = np.array(zvec.express_coordinates(self._csys0)).astype(float)
        projected_distance = delta_x.dot(zvec_0)
        if self._mode == 'factor':
            shift_amplitude = parameters[0]*projected_distance
        elif self._mode == 'max_delta':
            max_distance_reference = np.max(projected_distance[self._handle_nodes])
            factor = (max_distance_reference + parameters[0]) / max_distance_reference - 1.0
            shift_amplitude = factor*projected_distance
        else:
            raise AssertionError('mode is {}, but must be factor or maximum'.format(self._mode))
        shift = np.outer(shift_amplitude, zvec_0)

        delta_d[self._handle_nodes, 0] = delta_d[self._handle_nodes, 0] + shift[self._handle_nodes, 0]
        delta_d[self._handle_nodes, 1] = delta_d[self._handle_nodes, 1] + shift[self._handle_nodes, 1]
        delta_d[self._handle_nodes, 2] = delta_d[self._handle_nodes, 2] + shift[self._handle_nodes, 2]


class RadialStretchDeformationHandler(DeformationHandler):
    """
    RadialStretchDeformationHandler.

    Adds a parametric stretch into radial direction in the x-y-plane of a given sympy coordinate
    system.

    Examples
    --------

    >>> handle_nodes = np.array([0, 1, 2], dtype=np.int32)
    >>> x0 = np.random.rand(9).reshape(3,3)
    >>> d0 = np.zeros_like(x0)
    >>> delta_d = np.zeros_like(x0)
    >>> # Get initial coordinate system
    >>> csys_ref = study.csys
    >>> # Rotate 90 degree around initial x axis:
    >>> csys_new = csys_ref.locate_new('CenterPoint', 3*csys_ref.i + 4*csys_ref.j + 5*csys_ref.k)
    >>> # Create deformation handler
    >>> dh = RadialStretchDeformationHandler(handle_nodes, csys_ref, csys_new)
    >>> # Apply radial stretch of amplitude 2.0
    >>> dh.deform(np.array([2.0]), x0, d0, delta_d)

    """
    def __init__(self, handle_nodes, csys_ref, csys):
        """
        Parameters
        ----------
        handle_nodes : array_like
            array of dtype int32 containing the indices of the handle nodes
            that are to be set by the deformation handler
        csys_ref : CoordSys3D
            sympy coordinate system describing a reference system.
            Usually you pass study.csys as this argument.
        csys : CoordSys3D
            sympy coordinate system that is related to csys_ref.
            The z-axis of this system should point into the direction of the
            desired shift deformation.
        """
        super().__init__(handle_nodes)
        self._csys0 = csys_ref
        self._csys = csys

    def deform(self, parameters, x0, d0, delta_d):
        """
        Writes displacements into delta_d describing the deformation.

        Parameters
        ----------
        parameters : array_like
            Array dtype float containing the parameter values of the desired
            deformation. This handler expects just one parameter at
            parameters[0]. This parameter is the amplitude of the displacement.
        x0 : array_like
            Array dtype float containing the node coordinates of all nodes
            in reference configuration.
        d0 : array_like
            Array dtype float containing the nodal displacement at the
            beginning of the deformation
        delta_d : array_like
            Array to write the result into. The result is added to
            delta_d. Thus, you need to zero this array before you call this
            method, if you want the effect of the DeformationHandler
            only.

        Returns
        -------

        """
        center_coords = np.array(self._csys.origin.express_coordinates(self._csys0)).astype(float)
        sb_x = self._csys0.origin.locate_new('e_x', 1 * self._csys.i)
        sb_y = self._csys0.origin.locate_new('e_y', 1 * self._csys.j)
        sb_z = self._csys0.origin.locate_new('e_z', 1 * self._csys.k)
        b_x = np.array(sb_x.express_coordinates(self._csys0)).astype(float)
        b_y = np.array(sb_y.express_coordinates(self._csys0)).astype(float)

        e_x = np.array([1.0, 0.0, 0.0])
        e_y = np.array([0.0, 1.0, 0.0])
        e_z = np.array([0.0, 0.0, 1.0])

        x0_c = x0[self._handle_nodes, :] - center_coords
        x0_c_x = x0_c.dot(b_x)
        x0_c_y = x0_c.dot(b_y)
        r0_c = np.outer(x0_c_x, b_x) + np.outer(x0_c_y, b_y)
        r0_c_abs = np.sqrt(x0_c_x ** 2 + x0_c_y ** 2)
        e_r = r0_c / r0_c_abs.reshape(-1, 1)
        # mask numerically difficult radii
        tol = 1.e-14
        e_r[r0_c_abs < tol, :] = 0.0

        delta_r = (parameters[0] * r0_c_abs - r0_c_abs)

        delta_x = delta_r * e_r.dot(e_x)
        delta_y = delta_r * e_r.dot(e_y)
        delta_z = delta_r * e_r.dot(e_z)

        delta_d[self._handle_nodes, 0] = delta_d[self._handle_nodes, 0] + delta_x
        delta_d[self._handle_nodes, 1] = delta_d[self._handle_nodes, 1] + delta_y
        delta_d[self._handle_nodes, 2] = delta_d[self._handle_nodes, 2] + delta_z
