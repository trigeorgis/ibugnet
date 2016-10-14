from setuptools import setup, find_packages


setup(
    name='ibugnet',
    version='0.1',
    description='Dense Estimation Tasks for ibug use',
    author='George Trigeorgis',
    author_email='trigeorgis@gmail.com',
    packages=find_packages(),
    install_requires=['menpo>=0.7,<0.8', 'tensorflow>=0.10']
)
