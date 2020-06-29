""" Setup physlearn package."""

from __future__ import absolute_import

from setuptools import setup, find_packages


def setup_package():

    # with open('README', 'r') as f:
    #     long_description = f.read()

    setup(name='physlearn',
          version='0.1',
          license='MIT',
          description='A multi-target regression suite with a regressor dictionary.',
          #long_description=long_description,
          author='Alex Wozniakowski',
          author_email='wozn0001@e.ntu.edu.sg',
          packages=find_packages())

if __name__ == '__main__':
    setup_package()
