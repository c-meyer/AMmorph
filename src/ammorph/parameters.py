#
# Copyright (c) 2021 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
# AUTHOR: Christian Meyer
#

import sympy

class ParameterManager:
    _parameters = []
    _descriptions = {}
    _defaults = {}
    _eval_cache = None

    def __init__(self):
        return

    def add_parameter(self, name, description=None, default=0.0):
        new_parameter = sympy.symbols(name)
        if new_parameter in self._parameters:
            raise ValueError('Parameter {} is already defined.'.format(name))
        self._parameters.append(new_parameter)
        if description is None:
            description = name
        self._descriptions.update({name: description})
        self._defaults.update({name: default})
        self._eval_cache = sympy.lambdify(
            sympy.Matrix(self._parameters).transpose(),
            self._parameters, 'numpy')

    def evaluate(self, p):
        return self._eval_cache(p)

    def __str__(self):
        s = """
Parameter Manager
-----------------
Number of parameters: {}

Name       | Default     | Description
------------------------------------------------------------------------------
""".format(len(self._parameters))
        for p in self._descriptions.keys():
            pstring = "{:10s} | {:.3e}   | {:40s}\n".format(p,
                                                self._defaults[p],
                                                self._descriptions[p])
            s = s + pstring
        return s
