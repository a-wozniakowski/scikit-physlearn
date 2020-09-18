# Configuration file for the Sphinx documentation builder.

import datetime
import guzzle_sphinx_theme
import sphinx

from packaging.version import parse
from sphinx.errors import VersionRequirementError

# -- Project information -----------------------------------------------------

project = 'Scikit-physlearn'
copyright = '%s, Alex Wozniakowski (MIT License)' % str(datetime.datetime.now().year)
author = 'Alex Wozniakowski'

# -- General configuration ---------------------------------------------------

# Minimum version of sphinx required due to sphinx.ext.napoleon.
needs_sphinx = '1.3'
if needs_sphinx > sphinx.__version__:
    raise VersionRequirementError('The Sphinx version is less than the '
                                  'minimum version: %s.'
                                  % (needs_sphinx))

# Add any Sphinx extension module names here, as strings.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']

autodoc_default_options = {'members': True,
                           'inherited-members': True,
                           'show-inheritance': True,
                           'member-order': 'bysource',
                           'private-members': True,
                           'special-members': '__call__'}

autodoc_typehints = 'none'

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

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'
extensions.append("guzzle_sphinx_theme")
html_theme_options = {'project_nav_name': 'Scikit-physlearn'}

html_sidebars = {'**': ['logo-text.html', 'globaltoc.html', 'searchbox.html']}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
