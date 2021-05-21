import numpy as np

from ammorph.parameters import ParameterManager


def test_parameters_eval():
    pm = ParameterManager()
    pm.add_parameter('x')
    pm.add_parameter('y')
    desired = np.array([1.0, 3.0], dtype=np.float64)
    actual = pm.evaluate(desired)
    assert actual == desired


def test_parameters_print():
    pm = ParameterManager()
    pm.add_parameter('L', 'Length', 23.0)
    pm.add_parameter('W', 'Width', 4.5)
    pm.add_parameter('H', 'Height', 0.5)
    print(pm)
