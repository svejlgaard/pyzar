from setuptools import setup

setup(
  name = 'pyzar',
  url = 'https://github.com/svejlgaard/pyzar',
  author = 'Simone Vejlgaard Nielsen',
  author_email = 'fgj787@alumni.ku.dk',
  
  
  install_requires = ['numpy','pandas', 'pyspeckit','astropy','matplotlib','sklearn','scipy','extinction','datatime','json']
  version = '20.01'
  
  
  description = 'A package to detect spectral lines, determine redshift and measure equivalent width of QSO and QAL.'
  long_description = open('README.rst').read(),
  
  packages = ['qso']
)
