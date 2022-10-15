==================
Installation Guide
==================

There are different ways to install scikit-physlearn. They include:

- `Installing the latest stable release hosted on PyPI <#Installing-from-PyPI>`_.

- `Building the package from source <#Building-from-source>`_.


Installing from PyPI
====================

The scikit-physlearn pre-built binary wheel is available from Python
Package Index `(PyPI) <https://pypi.org/project/scikit-physlearn/>`_.
You may download and install it by running:

.. code-block:: bash

    pip install scikit-physlearn


Building from source
====================

Use `Git <https://git-scm.com/>`_ to clone the latest source from the scikit-physlearn
`repository <https://github.com/a-wozniakowski/scikit-physlearn>`_ on Github:

.. code-block:: bash

    git clone git://github.com/a-wozniakowski/scikit-physlearn.git
    cd scikit-physlearn

Then, build the project with ``pip`` in
`editable mode <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_:
    
.. code-block:: bash

    pip install -e

The global editable option ``-e`` invokes the setuptools develop mode:
``setup.py develop``. In other words, the build places a
``scikit_physlearn.egg-info`` directory adjacent to the code and resources.

Moreover, all of the dependencies are automatically managed by the build:

.. code-block:: bash

    numpy
    scipy
    scikit-learn
    pandas
    shap
    ipython
    bayesian-optimization
    catboost
    xgboost
    lightgbm
    mlxtend
    joblib
    threadpoolctl
    cython
    python-Levenshtein
