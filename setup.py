from setuptools import setup

setup(
    name='quantum-gates',
    version='2.4.0',
    author='M. Grossi, G. D. Bartolomeo, M. Vischi, P. Da Rold, R. Wixinger, N. Pacey, C. Christen',
    author_email='michele.grossi@cern.ch, dibartolomeo.giov@gmail.com, vischimichele@gmail.com, PAOLO.DAROLD@studenti.units.it, roman.wixinger@gmail.com, npacey01@gmail.com, cherilyn.christen@epfl.ch',
    packages=['quantum_gates',
              'quantum_gates._gates',
              'quantum_gates._legacy',
              'quantum_gates._simulation',
              'quantum_gates._qiskit_provider',
              'quantum_gates._utility'],
    license='MIT',
    python_requires='>=3.9'
)
