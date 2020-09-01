"""Setup physlearn package."""

# Author: Alex Wozniakowski
# Licence: MIT

import sys

from setuptools import setup, find_packages


DISTNAME = 'scikit-physlearn'
DESCRIPTION = 'A Python package for single-target and multi-target regression tasks.'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Alex Wozniakowski'
MAINTAINER_EMAIL = 'wozn0001@e.ntu.edu.sg'
URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
DOWNLOAD_URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
PROJECT_URLS = dict(Paper='https://arxiv.org/abs/2005.06194')

import physlearn
import physlearn._build_utils.min_dependencies as min_deps

VERSION = physlearn.__version__

SETUPTOOLS_COMMANDS = set(['develop', 'release', 'bdist_egg', 'bdist_rpm',
                           'bdist_wininst', 'install_egg_info', 'build_sphinx',
                           'egg_info', 'easy_install', 'upload', 'bdist_wheel',
                           '--single-version-externally-managed'])

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    extra_setuptools_args = dict(zip_safe=False,
                                 include_package_data=True,
                                 extras_require={key: min_deps.tag_to_packages[key]
                                                 for key in ['docs', 'tests']})
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


def setup_package():

    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type='text/x-rst',
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    license=LICENSE,
                    classifiers=CLASSIFIERS,
                    packages=PACKAGES,
                    package_data={'': ['*.json', '*.csv']},
                    install_requires=min_deps.tag_to_packages['install'],
                    **extra_setuptools_args)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
