""" Setup physlearn package."""

from __future__ import absolute_import

from setuptools import setup, find_packages


NAME = 'physlearn'
DESCRIPTION = 'A multi-target regression suite with a regressor dictionary.'
with open('README.md', 'r') as f:
        LONG_DESCRIPTION = f.read()
MAINTAINER = 'Alex Wozniakowski'
MAINTAINER_EMAIL = 'wozn0001@e.ntu.edu.sg'
URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
VERSION = '0.1'
LICENSE = 'MIT'
PACKAGES = find_packages()


def setup_package():

    metadata = dict(name=NAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    url=URL,
                    version=VERSION,
                    license=LICENSE,
                    packages=PACKAGES)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
