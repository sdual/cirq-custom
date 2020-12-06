from typing import Union, Tuple, List

import cirq
import numpy as np
import sympy
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


def append_su2(circuit: cirq.Circuit, qubit: cirq.ops.Qid, param1: value.TParamVal, param2: value.TParamVal,
               param3: value.TParamVal) -> cirq.Circuit:
    circuit.append(cirq.rz(param1).on(qubit))
    circuit.append(cirq.ry(param2).on(qubit))
    circuit.append(cirq.rz(param3).on(qubit))
    return circuit


def append_su4(circuit: cirq.Circuit, qubit1: cirq.ops.Qid, qubit2: cirq.ops.Qid, k1_params: List[value.TParamVal],
               k2_params: List[value.TParamVal], a_params: List[value.TParamVal]) -> cirq.Circuit:
    if len(k1_params) != 6 or len(k2_params) != 6 or len(a_params) != 3:
        raise ValueError('invalid params nums.')

    append_su2(circuit, qubit1, k1_params[0], k1_params[1], k1_params[2])
    append_su2(circuit, qubit2, k1_params[3], k1_params[4], k1_params[5])

    circuit.append(zz_rot(a_params[0]).on(qubit1, qubit2))
    circuit.append(yy_rot(a_params[1]).on(qubit1, qubit2))
    circuit.append(zz_rot(a_params[2]).on(qubit1, qubit2))

    append_su2(circuit, qubit1, k2_params[0], k2_params[1], k2_params[2])
    append_su2(circuit, qubit2, k2_params[3], k2_params[4], k2_params[5])
    return circuit
