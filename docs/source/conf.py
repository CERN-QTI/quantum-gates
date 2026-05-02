# Configuration file for the Sphinx documentation builder.

# -- Project information

import os
import sys
from importlib.metadata import version as _pkg_version, PackageNotFoundError

project = 'Quantum Gates'
copyright = '2023-2026, Di Bartolomeo, Vischi, Cesa, Grossi, Donadi, Bassi, Da Rold, Christen, Pacey, Crognaletti, Wixinger'
author = 'Di Bartolomeo, Vischi, Cesa, Grossi, Donadi, Bassi, Da Rold, Christen, Pacey, Crognaletti, Wixinger'

try:
    release = _pkg_version("quantum-gates")
except PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

sys.path.insert(0, os.path.abspath(os.path.join('../../src')))

# -- General configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.linkcode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = "both"
autodoc_typehints_format = "short"
napoleon_include_private_with_doc = True
napoleon_google_docstring = True

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    # Do something
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", autodoc_process_signature)


def linkcode_resolve(domain, info):
    """Adds a [source] button to the docs.
    """
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')

    source_lookup = {
        "quantum_gates/backends": "quantum_gates/_simulation/backend",
        "quantum_gates/gates": "quantum_gates/_gates/gates",
        "quantum_gates/circuits": "quantum_gates/_simulation/circuit",
        "quantum_gates/factories": "quantum_gates/_gates/factories",
        "quantum_gates/integrators": "quantum_gates/_gates/integrator",
        "quantum_gates/metrics": "quantum_gates/_utility/simulations_utility",
        "quantum_gates/pulses": "quantum_gates/_gates/pulse",
        "quantum_gates/quantum_algorithms": "quantum_gates/_utility/quantum_algorithms",
        "quantum_gates/simulators": "quantum_gates/_simulation/simulator",
        "quantum_gates/utilities": "quantum_gates/_utility/simulations_utility",
        "quantum_gates/qiskit_provider": "quantum_gates/_qiskit_provider/ng_provider",
    }

    # Replace filename if the source is hidden
    if filename in source_lookup:
        filename = source_lookup[filename]

    return f"https://github.com/CERN-QTI/quantum-gates/tree/main/src/{filename}.py"
