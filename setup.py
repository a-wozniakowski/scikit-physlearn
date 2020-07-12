""" Setup physlearn package."""

from setuptools import setup, find_packages


DISTNAME = 'scikit-physlearn'
DESCRIPTION = 'A Python package for single-target and multi-target regression tasks.'
MAINTAINER = 'Alex Wozniakowski'
MAINTAINER_EMAIL = 'wozn0001@e.ntu.edu.sg'
URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
DOWNLOAD_URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
VERSION = '0.1'
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


def setup_package():

    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description='Scikit-physlearn is a Python package for single-target and ' + \
                                     'multi-target regression. It is designed to amalgamate Scikit-learn, ' + \
                                     'LightGBM, XGBoost, CatBoost, and Mlxtend regressors into a unified ' + \
                                     'Regressor. It follows the Scikit-learnAPI, represents data in pandas, ' + \
                                     'and it supports base boosting.',
                    long_description_content_type='text/markdown',
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    license=LICENSE,
                    classifiers=CLASSIFIERS,
                    packages=PACKAGES,
                    package_data={'': ['*.json', '*.csv']})

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
