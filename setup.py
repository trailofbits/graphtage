import os
from setuptools import setup, find_packages

VERSION_MODULE_PATH = os.path.join(os.path.realpath(os.path.dirname(__file__)), "graphtage", "version.py")


def get_version_string():
    version = {}
    with open(VERSION_MODULE_PATH) as f:
        exec(f.read(), version)
    return version['VERSION_STRING']


setup(
    name='graphtage',
    description='A utility to diff tree-like files such as JSON and XML.',
    url='https://github.com/trailofbits/graphtage',
    author='Trail of Bits',
    version=get_version_string(),
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'colorama',
        'intervaltree',
        'PyYAML',
        'scipy>=1.4.0',
        'tqdm',
        'typing_extensions>=3.6.5'
    ],
    entry_points={
        'console_scripts': [
            'graphtage = graphtage.__main__:main'
        ]
    },
    extras_require={
        "dev": ["flake8", "Sphinx", "pytest", "sphinx_rtd_theme==0.4.3"]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Utilities'
    ]
)
