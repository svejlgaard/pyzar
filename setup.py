import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'pyzar',
  url = 'https://github.com/svejlgaard/pyzar',
  author = 'Simone Vejlgaard Nielsen',
  author_email = 'fgj787@alumni.ku.dk',
  
  
  install_requires = ['numpy','pandas', 'pyspeckit','astropy','matplotlib','sklearn','scipy','extinction','datatime','json']
  version = '20.01'
  
  
  description = 'A package to detect spectral lines, determine redshift and measure equivalent width of QSO and QAL.'
  long_description = long_description,
  long_description_content_type = "text/markdown",

  
  scripts = ['scripts/eqw.py'],
  packages = setuptools.find_packages(),
  
  python_requires = '>=3.6',
)
