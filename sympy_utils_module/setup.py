from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name = 'sympy_utils',
    version = '1.0.0',
    description = 'Utilities for SymPy',
    long_description = readme(),
    classifiers=[
        'Development Status :: Beta',
        'License :: GNU GPL v3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Mathematics, Engineering',
    ],
    keywords='sympy utils equation constant',
    url = 'https://github.com/Davide-sd/sym-comp-sympy',
    author = 'Davide Sandona',
    author_email = 'sandona.davide@gmail.com',
    license='GNU GPL v3',
    packages = [
        'sympy_utils',
    ],
    include_package_data=True,
    zip_safe = False,
    install_requires = [
        "numpy",
        "sympy>=1.6.1",
        "matplotlib",
        "mpmath",
        "graphviz",
    ]
)