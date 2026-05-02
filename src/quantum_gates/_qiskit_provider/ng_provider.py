from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from .ng_backend import NoisyGatesBackend


class NoisyGatesProvider:
    """Provider for backends for Noisy Gates model.
    At the moment is possible to use a backend that work alongside with IBM real device
    """

    def __init__(self, token=None, crn_instance = None):
        """

        Args:
            token (str): Token for the availability of real device from IBM. Defaults to None.
        """
        super().__init__()
        self.token = token
        self.crn_instance = crn_instance

        if token is not None and crn_instance is not None:
            self.qiskit_provider = QiskitRuntimeService(channel='ibm_quantum_platform', token=self.token, instance = crn_instance)
        else:
            print("Warning: Load fake backend because there was no token provided.")
            self.qiskit_provider = FakeProviderForBackendV2()

    def backends(self):
        return 

    def ibm_backends(self, **kwargs):
        """Show the possible usable device from IBM

        Raises:
            ValueError: If a token is not provided, it's not possibile to access to IBM device

        Returns:
            List: List of available device from IBM
        """
        if self.token is None:
            print("Warning: Load fake backend because there was no token provided.")
        return self.qiskit_provider.backends()

    def get_ibm_backend(self, name_backend, **kwargs) -> NoisyGatesBackend:
        """Choose one of the real device to simulare the features during the running process

        Args:
            name_backend (str): Name of the device from IBM, real if it's provided a token, fake if there
                isn't a token. Example: ``ibm_sherbrooke`` for a real device, ``fake_sherbrooke`` for a fake one.

        Raises:
            ValueError: If a token is not provided, it's not possibile to access to IBM device

        Returns:
            NoisyGatesBackend: Return a noisy gates backend that use ibm_device as real device from IBM
        """
        assert self.token or name_backend.startswith("fake"), "Token is required to access to IBM device."
        ibm_device = self.qiskit_provider.backend(name=name_backend)
        backend = NoisyGatesBackend(device=ibm_device)
        return backend
