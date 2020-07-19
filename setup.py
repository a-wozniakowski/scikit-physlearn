""" Setup physlearn package."""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext


DISTNAME = 'scikit-physlearn'
DESCRIPTION = 'A Python package for single-target and multi-target regression tasks.'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Alex Wozniakowski'
MAINTAINER_EMAIL = 'wozn0001@e.ntu.edu.sg'
URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
DOWNLOAD_URL = 'https://github.com/a-wozniakowski/scikit-physlearn'
PROJECT_URLS = dict(Paper='https://arxiv.org/abs/2005.06194')
VERSION = '0.1.3'
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

# We require ipython for SHAP, as display is used in force_plot.
# We require Levenshtein, as it is used for autocorrection.
# We restrict the version of XGBoost due to issue #1215 in SHAP.
REQUIRED = ['numpy', 'scipy', 'scikit-learn>=0.23.0', 'pandas',
            'shap', 'ipython', 'bayesian-optimization',
            'catboost', 'xgboost<1.1.0', 'lightgbm',
            'mlxtend', 'python-Levenshtein-wheels'],


# This class follows from SHAP for building C extensions.
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            setattr(__builtins__, "__NUMPY_SETUP__", False)
        import numpy
        print("numpy.get_include()", numpy.get_include())
        self.include_dirs.append(numpy.get_include())


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
                    cmdclass={'build_ext': build_ext},
                    setup_requires=['numpy'],
                    install_requires=REQUIRED,
                    zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
