"""Setup physlearn package."""

# Author: Alex Wozniakowski
# Licence: MIT

import os
import sys

from distutils.file_util import copy_file
from setuptools import setup, find_packages


DISTNAME = 'scikit-physlearn'
DESCRIPTION = 'A machine learning library for regression.'

MAINTAINER = 'Alex Wozniakowski'
MAINTAINER_EMAIL = 'wozn0001@e.ntu.edu.sg'
URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
DOWNLOAD_URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
PROJECT_URLS = dict(Paper='https://iopscience.iop.org/article/10.1088/2632-2153/ac1ee9')

SETUPTOOLS_COMMANDS = set(['develop', 'release', 'bdist_egg', 'bdist_rpm',
                           'bdist_wininst', 'install_egg_info', 'build_sphinx',
                           'egg_info', 'easy_install', 'upload', 'bdist_wheel',
                           '--single-version-externally-managed'])

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    extra_setuptools_args = dict(zip_safe=False,
                                 include_package_data=True,
                                 extras_require=dict(docs=['sphinx >= 3.0.3',
                                                           'sphinx-gallery >= 0.7.0']))
else:
    extra_setuptools_args = dict()

LICENSE = 'MIT'
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
PACKAGES = find_packages()

INSTALL_REQUIRES = ['matplotlib <= 3.2.0',
                    'numpy',
                    'scipy',
                    'scikit-learn >= 0.24.1',
                    'pandas >= 1.0.0',
                    'shap >= 0.36.0',
                    'ipython >= 7.11.0',
                    'bayesian-optimization >= 1.2.0',
                    'catboost >= 0.23.2',
                    'xgboost >= 1.4.1',
                    'lightgbm >= 2.3.0',
                    'mlxtend >= 0.17.0',
                    'python-levenshtein-wheels >= 0.13.1']


if __name__ == '__main__':
    CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    if os.path.isfile(os.path.join(CURRENT_DIR, 'VERSION.txt')):
        copy_file(os.path.join(CURRENT_DIR, 'VERSION.txt'),
                  os.path.join(CURRENT_DIR, 'physlearn', 'VERSION.txt'),
                  verbose=0)
    VERSION = open(os.path.join(CURRENT_DIR, 'physlearn', 'VERSION.txt'), encoding='utf-8').read().strip()
    LONG_DESCRIPTION = open(os.path.join(CURRENT_DIR, 'README.md'), encoding='utf-8').read()

    sys.path.insert(0, CURRENT_DIR)

    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type='text/markdown',
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    license=LICENSE,
                    classifiers=CLASSIFIERS,
                    packages=PACKAGES,
                    package_data={'': ['*.json', '*.csv']},
                    install_requires=INSTALL_REQUIRES,
                    **extra_setuptools_args)

    setup(**metadata)
