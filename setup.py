from setuptools import setup

setup(
    name='quantum-gates',
    version='2.4.0',
    author='G. Di Bartolomeo, M. Vischi, F. Cesa, M. Grossi, S. Donadi, A. Bassi, P. Da Rold, C. Christen, N. Pacey, G. Crognaletti, R. Wixinger',
    author_email='dibartolomeo.giov@gmail.com, vischimichele@gmail.com, michele.grossi@cern.ch, paolo.darold@studenti.units.it, cherilyn.christen@epfl.ch, npacey01@gmail.com, giulio.crognaletti@phd.units.it, roman.wixinger@gmail.com',
    packages=['quantum_gates',
              'quantum_gates._gates',
              'quantum_gates._legacy',
              'quantum_gates._simulation',
              'quantum_gates._qiskit_provider',
              'quantum_gates._utility'],
    license='MIT',
    python_requires='>=3.9'
)
