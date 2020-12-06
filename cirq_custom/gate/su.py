from typing import Union, List, Tuple

import numpy as np
import sympy
import cirq
from cirq import value, protocols
from cirq.ops import gate_features, eigen_gate
from cirq.ops.eigen_gate import EigenComponent

from cirq_custom.gate.matrix import XX, YY, ZZ


class XXRot(eigen_gate.EigenGate, gate_features.TwoQubitGate):

    def _eigen_components(self) -> List[Union[EigenComponent,
                                              Tuple[float, np.ndarray]]]:
        return [
            (0, (np.eye(4) + XX) / 2.0),
            (1, (np.eye(4) - XX) / 2.0)]


class YYRot(eigen_gate.EigenGate, gate_features.TwoQubitGate):

    def _eigen_components(self) -> List[Union[EigenComponent,
                                              Tuple[float, np.ndarray]]]:
        return [
            (0, (np.eye(4) + YY) / 2.0),
            (1, (np.eye(4) - YY) / 2.0)]


class ZZRot(eigen_gate.EigenGate, gate_features.TwoQubitGate):

    def _eigen_components(self) -> List[Union[EigenComponent,
                                              Tuple[float, np.ndarray]]]:
        return [
            (0, (np.eye(4) + ZZ) / 2.0),
            (1, (np.eye(4) - ZZ) / 2.0)]


def xx_rot(rads: value.TParamVal) -> XXRot:
    pi = sympy.pi if protocols.is_parameterized(rads) else np.pi
    return XXRot(exponent=rads / pi, global_shift=-0.5)


def yy_rot(rads: value.TParamVal) -> YYRot:
    pi = sympy.pi if protocols.is_parameterized(rads) else np.pi
    return YYRot(exponent=rads / pi, global_shift=-0.5)


def zz_rot(rads: value.TParamVal) -> ZZRot:
    pi = sympy.pi if protocols.is_parameterized(rads) else np.pi
    return ZZRot(exponent=rads / pi, global_shift=-0.5)

def append_su4():
    cirq