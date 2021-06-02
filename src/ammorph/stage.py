#
# Copyright (c) 2021 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
# AUTHOR: Christian Meyer
#

import logging
import numpy as np
from scipy.linalg import solve
import sympy
from sympy import lambdify

from ammorph.assembly import assemble
from ammorph.evaluation import evaluate


class CompositeStage:
    """
    CompositeStage describes a stage that is composed of substages.

    The difference between a normal stage that is added to
    a study and a CompositeStage is that the children stages of the
    CompositeStage do not use the current deformation as reference coordinates
    for interpolation.
    Instead each substage uses the deformation at the beginning
    of the stage.
    """
    def __init__(self, callback=None, callbackargs=(), name=None):
        """
        Parameters
        ----------
        callback : function, optional
            Callback function with signature callback(x, d, delta_d, \*args)
            This function is called after the stage has been finished.
            x are the reference coordinates
            d is the displacement vector at beginning of the stage
            delta_d is the difference to the displacement vector after
            processing the stage.
        callbackargs : tuple, optional
            arguments that are passed as \*args in the callback function
        name : str, optional
            A name can be passed that is used in logging features to identify
            the stage that produces certain loggin messages.

        """
        self._callback = callback
        self._callbackargs = callbackargs
        if name is None:
            name = 'unnamed'
        self._name = name
        self._stages = []

    def add_child(self, stage, partitioned=False):
        """
        Adds a stage to the CompositeStage

        Parameters
        ----------
        stage : { ammorph.stage.Stage, ammorph.stage.CompositeStage }
            Stage object that is added to the composite.
        partitioned : bool
            Flag if child should be performed in partitioned mode or not.

        Returns
        -------

        """
        self._stages.append((stage, partitioned))

    def morph(self, p_args, p_vals, x0, d0, delta_d, partition=None,
              no_of_partitions=0):
        """
        Computes delta_d for the stage.

        Parameters
        ----------
        p_args : tuple
            tuple of sympy sympols or ammorph.parameters.Parameter objects
            that determines the meaning of the coordinates in the p_vals
            vector.
        p_vals : array_like
            Array of dtype float64 that describes the values of the paramters
            with which the delta_d is computed.
        x0 : array_like
            Array of dtype float64 describing the node coordinates of the
            nodes in reference configuration
        d0 : array_like
            Array of dtype float64 describing the displacement vector that
            has been computed in previous stages such that the current
            configuration before this stage is x0 + d0
        delta_d : array_like
            Array of dtype float64 used to write the result of the stage
            deformation into
        partition : array_like, optional
            Array of dtype int32 describing a partitioning of the deformation
            process. The value of each coordinate describes which partition
            is assigned to the corresponding node. If such an array is passed,
            the stage is performed in partitioned mode. Otherwise, no
            partitioning is done.
            Example: np.array([0, 3, 0, 5], dtype=np.int32) means that
            node indices 0 and 2 are assigned to partition 0
            node index 1 is assigned to partition 3 and
            node index 3 is assigned to partition 5.
        no_of_partitions : int, optional
            If the number of partitions is known a priori, this information
            can be passed to avoid this computation within the function

        Returns
        -------

        """

        logging.info('This is stage {}'.format(self._name))
        no_of_stages = len(self._stages)
        logging.info('-'*80 +
                     '\nRun {} substages ...\n'.format(no_of_stages))
        for i, (stage, partitioned) in enumerate(self._stages):

            logging.info('-'*80 +
                         '\nRunning Substage {} of {} ...\n'.format(i+1,
                                                                 no_of_stages))

            stage.morph(p_args, p_vals, x0, d0, delta_d, partition)
        if self._callback is not None:
            self._callback(x0, d0, delta_d, *self._callbackargs)
        logging.info('Finished {} substages.\n'.format(no_of_stages) +'-'*80 +
                     '\n')


class Stage:
    """
    Class to define a so called stage of a mesh deformation.

    A Stage can be seen as a deformation step. A full mesh morphing study
    consists of several stages that are applied in seqence to the mesh.
    """
    def __init__(self, active_nodes, fixed_nodes, rbf_func,
                 polynomial_order=None,
                 callback=None, callbackargs=(), name=None):
        r"""
        Parameters
        ----------
        active_nodes : array_like
            numpy array of dtype int32 containing the indices of all active
            nodes of this stage. Active nodes are those nodes that will have
            an updated deformation vector after the stage has been finished.
        fixed_nodes : array_like
            numpy array of dtype int32 containing the indices of all fixed
            nodes of this stage. Fixed nodes are nodes that are considered to
            be fixed during the deformation. Typically these nodes are at
            boundaries or nodes of special geometries like bore holes
            mounting geometries etc. which may not be deformed.
        rbf_func : str
            A string describing the Radial basis function that is used for
            the interpolation in this stage. The string must contain the
            character r as variable for the radial basis function phi(r).
            Valid strings can be obtained by using the static methods of the
            class :py:class:`ammorph.rbf_string.RBF`
        polynomial_order : {int, 0, 1}, optional
            Default: None
            Determines if the system of equations to determine the weights
            is augmented by polynomial terms. None adds no terms, 0 adds a
            constant term and 1 adds constant term and linear terms in x, y, z.
        callback : function, optional
            Callback function with signature callback(x, d, delta_d, \*args)
            This function is called after the stage has been finished.
            x are the reference coordinates
            d is the displacement vector at beginning of the stage
            delta_d is the difference to the displacement vector after
            processing the stage.
        callbackargs : tuple, optional
            arguments that are passed as \*args in the callback function
        name : str, optional
            A name can be passed that is used in logging features to identify
            the stage that produces certain loggin messages.
        """
        self._deformation_handlers = []
        self._active_nodes = active_nodes.astype(np.int32)
        self._fixed_nodes = fixed_nodes.astype(np.int32)
        self._fixed_and_handle_nodes = np.copy(self._fixed_nodes)
        self._no_of_fixed_and_handle_nodes = len(self._fixed_and_handle_nodes)
        self._rbffunc = rbf_func
        self._stages = []
        self._callback = callback
        self._callbackargs = callbackargs
        if name is None:
            name = 'unnamed'
        self._name = name
        spatial_dimension = 3
        if polynomial_order is None:
            extra_columns = 0
        elif polynomial_order == 0:
            extra_columns = 1
        elif polynomial_order == 1:
            extra_columns = spatial_dimension + 1
        else:
            raise NotImplementedError('Currently only augmentation until '
                                      'polynomial order = 1 is supported')
        self._polynomial_order = polynomial_order
        self._extra_columns = extra_columns

    def _morph_part1(self, p_args, p_vals, x0, d0, delta_d_handler):
        no_of_deformation_handlers = len(self._deformation_handlers)
        for i, (handler, parameters) in enumerate(self._deformation_handlers):
            logging.info('Call Deformation '
                         'Handler {} '
                         'of {} ...'.format(i+1, no_of_deformation_handlers))
            p = lambdify([p_arg for p_arg in p_args],
                         [pm for pm in parameters])
            handler.deform(p(*p_vals), x0, d0, delta_d_handler)

    @staticmethod
    def _get_partitioned_nodes(nodes, partition, i):
        return nodes[partition[nodes] == i]

    def _assemble_and_evaluate(self, x0, d0, active_nodes,
                               fixed_and_handle_nodes,
                               no_of_fixed_and_handle_nodes,
                               delta_d_handler, delta_d):
        # 2. Assemble m
        logging.info('Assemble interpolation matrix...')
        m = assemble(self._rbffunc, x0[fixed_and_handle_nodes, :] +
                     d0[fixed_and_handle_nodes, :],
                     self._polynomial_order)
        logging.info('Solve weights for interpolation...')
        if self._extra_columns == 0:
            gamma = solve(m, delta_d_handler[fixed_and_handle_nodes, :])
        else:
            gamma = solve(m, np.concatenate(
                (delta_d_handler[fixed_and_handle_nodes, :],
                 np.zeros((self._extra_columns, 3), dtype=np.float64)),
                axis=0))
        # 3. Evaluate
        logging.info('Evaluate interpolation...')
        s = evaluate(self._rbffunc, x0[active_nodes, :] +
                     d0[active_nodes, :],
                     x0[fixed_and_handle_nodes, :] +
                     d0[fixed_and_handle_nodes, :],
                     gamma[:no_of_fixed_and_handle_nodes],
                     gamma[no_of_fixed_and_handle_nodes:])

        delta_d[active_nodes, :] = s

    def _morph_partitioned(self, p_args, p_vals, x0, d0, delta_d, partition,
                           no_of_partitions):
        delta_d_handler= np.copy(delta_d)
        self._morph_part1(p_args, p_vals, x0, d0, delta_d_handler)

        for i in range(no_of_partitions):
            logging.info('--- Morph partition {} of {}'.format(
                i+1, no_of_partitions))
            active_nodes = self._get_partitioned_nodes(self._active_nodes,
                                                       partition, i)
            fixed_and_handle_nodes = self._get_partitioned_nodes(
                self._fixed_and_handle_nodes, partition, i)
            no_of_fixed_and_handle_nodes = len(fixed_and_handle_nodes)
            self._assemble_and_evaluate(x0, d0, active_nodes,
                                        fixed_and_handle_nodes,
                                        no_of_fixed_and_handle_nodes,
                                        delta_d_handler, delta_d)
        del delta_d_handler

    def _morph_unpartitioned(self, p_args, p_vals, x0, d0, delta_d):
        delta_d_handler= np.copy(delta_d)
        self._morph_part1(p_args, p_vals, x0, d0, delta_d_handler)

        active_nodes = self._active_nodes
        fixed_and_handle_nodes = self._fixed_and_handle_nodes
        no_of_fixed_and_handle_nodes = len(fixed_and_handle_nodes)

        self._assemble_and_evaluate(x0, d0, active_nodes,
                                   fixed_and_handle_nodes,
                                   no_of_fixed_and_handle_nodes,
                                   delta_d_handler, delta_d)
        del delta_d_handler

    def add_deformation_handler(self, handler, parameters):
        """
        Adds a deformation handler object to the stage.

        Parameters
        ----------
        handler : ammorph.handler.DeformationHandler
            Deformation handler object that is added to the stage.
        parameters : {tuple, list}
            Tuple or list containing sympy expressions to describe the
            parameters of the deformation handler. The number and meaning
            of required expressions is described in the documentation for
            the DeformationHandler object that is wanted to be added.

        Returns
        -------

        Examples
        --------
        >>> import numpy as np
        >>> from ammorph import Stage, RBF
        >>> from ammorph.handler import ShiftDeformationHandler
        >>> rbf_func = RBF.thin_plate_spline(0.3)
        >>> active_nodes = np.array([0, 1, 2, 3], dtype=np.int32)
        >>> fixed_nodes = np.array([0], dtype=np.int32)
        >>> stage = Stage(active_nodes, fixed_nodes, rbf_func)
        >>> p = ammorph.Parameter('p')
        >>> expr = 3*p
        >>> csys = stage.csys
        >>> handler = ShiftDeformationHandler(handle_nodes, csys, csys)
        >>> stage.add_deformation_handler(handler, (expr, ))


        """
        self._deformation_handlers.append((handler, parameters))
        self._fixed_nodes = np.setdiff1d(self._fixed_nodes,
                                         handler.handle_nodes)
        self._fixed_and_handle_nodes = np.array([], dtype=np.int32)
        for d in self._deformation_handlers:
            self._fixed_and_handle_nodes = np.union1d(
                self._fixed_and_handle_nodes, d[0].handle_nodes)
        self._fixed_and_handle_nodes = np.union1d(self._fixed_and_handle_nodes,
                                                  self._fixed_nodes)
        self._no_of_fixed_and_handle_nodes = len(self._fixed_and_handle_nodes)

    def morph(self, p_args, p_vals, x0, d0, delta_d, partition=None,
              no_of_partitions=0):
        """
        Computes delta_d for the stage.

        Parameters
        ----------
        p_args : tuple
            tuple of sympy sympols or ammorph.parameters.Parameter objects
            that determines the meaning of the coordinates in the p_vals
            vector.
        p_vals : array_like
            Array of dtype float64 that describes the values of the paramters
            with which the delta_d is computed.
        x0 : array_like
            Array of dtype float64 describing the node coordinates of the
            nodes in reference configuration
        d0 : array_like
            Array of dtype float64 describing the displacement vector that
            has been computed in previous stages such that the current
            configuration before this stage is x0 + d0
        delta_d : array_like
            Array of dtype float64 used to write the result of the stage
            deformation into
        partition : array_like, optional
            Array of dtype int32 describing a partitioning of the deformation
            process. The value of each coordinate describes which partition
            is assigned to the corresponding node. If such an array is passed,
            the stage is performed in partitioned mode. Otherwise, no
            partitioning is done.
            Example: np.array([0, 3, 0, 5], dtype=np.int32) means that
            node indices 0 and 2 are assigned to partition 0
            node index 1 is assigned to partition 3 and
            node index 3 is assigned to partition 5.
        no_of_partitions : int, optional
            If the number of partitions is known a priori, this information
            can be passed to avoid this computation within the function

        Returns
        -------

        """
        logging.info('Morph stage {}'.format(self._name))
        if partition is None:
            self._morph_unpartitioned(p_args, p_vals, x0, d0, delta_d)
        else:
            if no_of_partitions == 0:
                no_of_partitions = len(np.unique(partition))
            self._morph_partitioned(p_args, p_vals, x0, d0, delta_d, partition,
                                    no_of_partitions)
        if self._callback is not None:
            self._callback(x0, d0, delta_d, *self._callbackargs)
