# Configuration file for the Sphinx documentation builder.

import sphinx
import sphinx_theme

from packaging.version import parse
from sphinx.errors import VersionRequirementError

# -- Project information -----------------------------------------------------

project = 'Scikit-physlearn'
copyright = '2020, Alex Wozniakowski (MIT License)'
author = 'Alex Wozniakowski'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
min_version = '1.3'  # Due to sphinx.ext.napoleon
if min_version > sphinx.__version__:
    raise VersionRequirementError('The Sphinx version is less than the minimum version: %s'
                                  % (min_version))

# Add any Sphinx extension module names here, as strings.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.napoleon',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.imgconverter']

autodoc_default_options = {'members': True,
                           'inherited-members': True,
                           'show-inheritance': True,
                           'member-order': 'bysource',
                           'private-members': True,
                           'special-members': '__call__'}

autodoc_typehints = 'description'

autosummary_generate = True

import physlearn
parsed_version = parse(physlearn.__version__)
version = ".".join(parsed_version.base_version.split(".")[:2])
# The full version, including alpha/beta/rc tags.
# It removes post from release name.
if parsed_version.is_postrelease:
    release = parsed_version.base_version
else:
    release = physlearn.__version__

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'stanford_theme'
html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
