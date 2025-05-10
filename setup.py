from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.1'
DESCRIPTION = 'Solve and estimate linearized DSGE models'

# Setting up
setup(
    name="dsgepy",
    version=VERSION,
    author="Gustavo Amarante",
    author_email="gusamarante@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'sympy',
        'tqdm',

    ],
    keywords=['dsge', 'macroeconomics'],
)
