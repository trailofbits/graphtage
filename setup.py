import os
from setuptools import setup, find_packages

HERE = os.path.realpath(os.path.dirname(__file__))

VERSION_MODULE_PATH = os.path.join(HERE, "graphtage", "version.py")
README_PATH = os.path.join(HERE, "README.md")


def get_version_string():
    version = {}
    with open(VERSION_MODULE_PATH) as f:
        exec(f.read(), version)
    return version['VERSION_STRING']


def get_readme():
    with open(README_PATH, encoding='utf-8') as f:
        return f.read()


setup(
    name='graphtage',
    description='A utility to diff tree-like files such as JSON and XML.',
    license="LGPL-3.0-or-later",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/trailofbits/graphtage',
    project_urls={
        'Documentation': 'https://trailofbits.github.io/graphtage',
        'Source': 'https://github.com/trailofbits/graphtage',
        'Tracker': 'https://github.com/trailofbits/graphtage/issues',
    },
    author='Trail of Bits',
    version=get_version_string(),
    packages=find_packages(exclude=['test']),
    python_requires='>=3.6',
    install_requires=[
        'colorama',
        'intervaltree',
        'json5==0.9.5',
        'numpy>=1.19.4',
        'PyYAML',
        'scipy>=1.4.0',
        'tqdm',
        'typing_extensions>=3.7.4.3'
    ],
    entry_points={
        'console_scripts': [
            'graphtage = graphtage.__main__:main'
        ]
    },
    extras_require={
        "dev": ["flake8", "Sphinx", "pytest", "sphinx_rtd_theme==0.4.3", "twine"]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Utilities'
    ],
    include_package_data=True
)
