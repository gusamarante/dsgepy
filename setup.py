from setuptools import setup, find_packages

setup(
    name = 'pydsge',
    version = '0.1',
    author = 'Gustavo Amarante',
    author_email = 'developer@pydsge.com',
    description = ('This is a Python library to calibrate, estimate and analyze linearized DSGE models.'),
    license = 'BSD',
    keywords = 'dsge bayesian',
    url = 'http://pydsge.com/',
    packages = find_packages(),
    classifiers = [
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: BSD License',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires = [
      'numpy',
      'scipy',
      'tqdm',
      'pandas',
      'matplotlib',
      'sympy',
      'tables'
    ],
    tests_require = [
      'nose',
    ],
    extras_require = {
        'docs': [
          'Sphinx',
          'numpydoc',
        ],
        'tests': [
          'nose',
        ],
    },
)
