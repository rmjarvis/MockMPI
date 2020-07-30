from setuptools import setup, find_packages
import pathlib
import re

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Only uses numpy and multiprocessing
dependencies = ['numpy']

# Read in the version from mock_mpi/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file = (here / 'mock_mpi' / '_version.py')
verstrline = version_file.read_text(encoding='utf-8')
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
version = mo.group(1)
print('mock_mpi version is %s'%(version))

setup(
    name='MockMPI',
    version=version,
    description='A tool for mocking mpi4py for testing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rmjarvis/MockMPI',
    download_url='https://github.com/rmjarvis/MockMPI/releases/tag/v%s.zip'%version,
    author='Mike Jarvis and collaborators',
    author_email='michael@jarvis.net',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='MPI, development, unittesting',
    packages=['mock_mpi'],
    python_requires='>=3.5',
    install_requires=dependencies,
)
