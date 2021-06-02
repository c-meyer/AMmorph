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
from sympy.vector import CoordSys3D

class Study:
    """
    Study class to handle mesh morphing. This class is the starting point
    to define a mesh morphing problem.

    """
    def __init__(self, x0=None, copy=True, partition=None):
        """

        Parameters
        ----------
        x0 : array_like, optional
            Array dtype float64 with shape (number_of_nodes, 3)
            where the first column contains x coordinates, second column
            the y coordinates and the last column the z coordinates.
        copy : bool
            flag if x0 should be copied into the study object.
            default is True. Otherwise x0 is stored by reference.
        partition : array_like, optional
            Studies that have many boundary nodes often cannot be performed
            without a division into smaller problems. The partition paramter
            allows to pass mesh partionning information for this purpose.
            The argument is an array of dtype int32 with the length that is
            equal to the number of nodes in x0. The coordinates of this array
            contain the partition number that is assigned to the corresponding
            node. Example: np.array([0, 1, 1, 0], dtype=np.int)
            assigns partition 0 to nodes 0 and 3 and partition 1 to nodes
            1 and 2.
        """
        self._x0 = np.zeros((0, 3), dtype=np.float64)
        self._d = np.zeros((0, 3), dtype=np.float64)
        self._no_of_nodes = 0
        self._csys_base = CoordSys3D('E')
        self._stages = []
        self._partition = partition
        if partition is not None:
            self._no_of_partitions = len(np.unique(partition))
        else:
            self._no_of_partitions = 0
        if x0 is not None:
            self.set_x0(x0, copy)

    def set_x0(self, x0, copy=True):
        """
        Sets new reference coordinates to the study.

        Parameters
        ----------
        x0 : array_like, optional
            Array dtype float64 with shape (number_of_nodes, 3)
            where the first column contains x coordinates, second column
            the y coordinates and the last column the z coordinates.
        copy : bool
            flag if x0 should be copied into the study object.
            default is True. Otherwise x0 is stored by reference.

        Returns
        -------

        """
        if copy:
            self._x0 = np.copy(x0)
        else:
            self._x0 = x0
        self._no_of_nodes = self._x0.shape[0]
        self._d = np.zeros_like(self._x0)
        self._delta_d = np.zeros_like(self._x0)

    @property
    def no_of_nodes(self):
        """
        Returns the number of nodes in the study.

        Returns
        -------
        no_of_nodes : int
            The number of nodes.
        """
        return self._no_of_nodes

    @property
    def csys(self):
        """
        Returns a sympy.CoordSys3D object for a Euclidean Coordinate System
        that is used as reference system. A reference to this object is often
        needed for the instantiation of DeformationHandlers that need
        information about a reference system they can relate their local
        coordinate system to.

        Returns
        -------
        csys : CoordSys3D
            The reference coordinate system object.
        """
        return self._csys_base

    def add_stage(self, stage, partitioned=False):
        """
        Add a stage to the study.

        Parameters
        ----------
        stage : { ammorph.stage.Stage, ammorph.stage.CompositeStage }
            stage object that is added.
        partitioned : bool
            Flag that determines if this stage shall be performed in
            partitioned mode or not. Default: False

        Returns
        -------

        """
        self._stages.append((stage, partitioned))

    def morph(self, p_args, p_vals, x_out):
        """
        Run the study.

        Parameters
        ----------
        p_args : { tuple, list }
            list with sympy symbols or ammorph.Paramter objects that define
            the order and meaning of p_vals argument.
        p_vals : array_like
            array of dtype float that define the values of the parameters
            specified in p_args.
        x_out : array_like
            array of dtype float64 to write the result of the morphin process.

        Returns
        -------

        """
        x_out[:, :] = self._x0[:, :]
        no_of_stages = len(self._stages)
        self._d[:, :] = 0.0
        delta_d = np.zeros_like(self._d)
        for i, (stage, partitioned) in enumerate(self._stages):

            logging.info('#'*80 +
                         '\nRunning stage {} of {} ...\n'.format(i+1,
                                                               no_of_stages))
            delta_d[:, :] = 0.0
            if partitioned:
                stage.morph(p_args, p_vals, self._x0, self._d, delta_d,
                            partition=self._partition,
                            no_of_partitions=self._no_of_partitions)
            else:
                stage.morph(p_args, p_vals, self._x0, self._d, delta_d)
            self._d[:, :] += delta_d
        x_out[:, :] = self._x0 + self._d
