from src.quantum_gates.utilities import DeviceParameters
from tests.helpers.paths import device_parameters_path


qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]
location = device_parameters_path("ibm_kyoto")
device_param = DeviceParameters(qubits_layout=qubits_layout)
device_param.load_from_texts(location)
T1, T2, p, rout, p_int, t_int, tm, dt, metadata = device_param.get_as_tuple()


INT_args = {
    "t_int": t_int[1][0],
    "p_i_k": p_int[1][0],
    "p_i": p[0],
    "p_k": p[1],
    "T1_ctr": T1[0],
    "T2_ctr": T2[0],
    "T1_trg": T1[1],
    "T2_trg": T2[1]
}
