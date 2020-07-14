""" Setup physlearn package."""

from setuptools import setup, find_packages


DISTNAME = 'scikit-physlearn'
DESCRIPTION = 'A Python package for single-target and multi-target regression tasks.'

# Convert Markdown file into RestructuredText.
# Use the README without images.
try:
    import pypandoc
    LONG_DESCRIPTION = pypandoc.convert('pypi/README.md', 'rst')
except (IOError, ImportError):
    LONG_DESCRIPTION = open('pypi/README.md').read()

MAINTAINER = 'Alex Wozniakowski'
MAINTAINER_EMAIL = 'wozn0001@e.ntu.edu.sg'
URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
DOWNLOAD_URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
VERSION = '0.1.2'
LICENSE = 'MIT'
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
PACKAGES = find_packages()

with open('requirements.txt') as f:
    REQUIRED = f.read().splitlines()


def setup_package():

    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type='text/markdown',
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    license=LICENSE,
                    classifiers=CLASSIFIERS,
                    packages=PACKAGES,
                    package_data={'': ['*.json', '*.csv']},
                    install_requires=REQUIRED)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
