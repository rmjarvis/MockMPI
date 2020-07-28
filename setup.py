from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='mock_mpi',
    version='0.5.0',
    description='A tool for mocking mpi4py for testing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/XXX/mock_mpi',
    author='Mike Jarvis',
    author_email='michael@jarvis.net',
    classifiers=[
        'Development Status :: 3 - Beta',
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
    install_requires=['numpy'],
)