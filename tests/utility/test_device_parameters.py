import pytest
import numpy as np

from src.quantum_gates.utilities import DeviceParameters
from tests.helpers.paths import device_parameters_path


location = device_parameters_path("ibm_kyiv")
invalid_location = 'invalid_location'
qubits_layout = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_device_parameters_load_from_json():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_json(location=location)


def test_device_parameters_load_from_texts():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_texts(location=location)


def test_load_device_parameters_is_equal():
    device_param_text = DeviceParameters(qubits_layout)
    device_param_text.load_from_texts(location=location)
    device_param_json = DeviceParameters(qubits_layout)
    device_param_json.load_from_json(location=location)
    assert device_param_text == device_param_json


def test_device_parameters_load_from_texts_with_invalid_input():
    device_param = DeviceParameters(qubits_layout)
    with pytest.raises(FileNotFoundError):
        device_param.load_from_texts(location=invalid_location)


def test_device_parameters_get_as_tuple():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_texts(location=location)
    T1, T2, p, rout, p_int, t_int, tm, dt, metadata = device_param.get_as_tuple()


def test_device_parameters_get_as_tuple_type():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_texts(location=location)
    items = device_param.get_as_tuple()
    assert all((isinstance(item, np.ndarray) or isinstance(item, dict) for item in items)), \
        f"Expected to get a tuple of numpy arrays but found {(type(item) for item in items)}."


def test_device_parameters_get_as_tuple_sizes():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_texts(location=location)
    T1, T2, p, rout, p_int, t_int, tm, dt, metadata = device_param.get_as_tuple()
    n = len(qubits_layout)
    assert T1.shape == (n,), f"T1 had invalid shape {T1.shape} instead of ({n})."
    assert T2.shape == (n,), f"T2 had invalid shape {T2.shape} instead of ({n})."
    assert p.shape == (n,), f"p had invalid shape {p.shape} instead of ({n})."
    assert rout.shape == (n,), f"rout had invalid shape {rout.shape} instead of ({n})."
    assert p_int.shape == (n, n), f"p_int had invalid shape {p_int.shape} instead of ({n}, {n})."
    assert t_int.shape == (n, n), f"t_int had invalid shape {t_int.shape} instead of ({n}, {n})."
    assert tm.shape == (n,), f"tm had invalid shape {tm.shape} instead of ({n})"
    assert dt.shape == (1,), f"dt had invalid shape {dt.shape} instead of ({n})"
    assert isinstance(metadata, dict), f"Expected metadata to be of type dict, but found {type(metadata)}."


def test_device_parameters_is_complete():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_texts(location=location)
    assert device_param.is_complete()


def test_device_parameters_is_complete_while_it_is_not():
    device_param = DeviceParameters(qubits_layout)
    assert not device_param.is_complete()


@pytest.mark.skip(reason="There are actually bad qubits in the dataset.")
def test_device_parameters_check_T1_and_T2_times():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_json(location=location)
    assert device_param.check_T1_and_T2_times(do_raise_exception=False)


def test_device_parameters_check_T1_and_T2_times_bad():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_json(location=location)
    device_param.T2 = np.zeros_like(device_param.T2)
    assert not device_param.check_T1_and_T2_times(do_raise_exception=False)


def test_device_parameters_check_T1_and_T2_times_bad_with_exception():
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_json(location=location)
    device_param.T2 = np.zeros_like(device_param.T2)
    with pytest.raises(Exception):
        device_param.check_T1_and_T2_times(do_raise_exception=True)
