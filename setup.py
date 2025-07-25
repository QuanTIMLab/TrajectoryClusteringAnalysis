import os
from setuptools import find_packages, setup, Extension
from typing import List
#from Cython.Build import cythonize
import numpy as np


HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Cette fonction retourne la liste des dépendances
    à partir d'un fichier requirements.txt.
    """
    with open(file_path) as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

def get_version():
    version_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'src',
        'trajectoryclusteringanalysis',
        '__version__.py'
    )
    version_ns = {}
    with open(version_path, 'r') as f:
        exec(f.read(), version_ns)
    return version_ns['__version__']

# Configuration des arguments de compilation par plateforme
if os.name == 'nt':  # Windows
    compile_args = ['/O2']  # Optimisation pour MSVC
    link_args = []
else:  # Linux/Mac
    compile_args = ['-O3', '-ffast-math']
    link_args = ['-O3']

# Définir l'extension Cython
extensions = [
    Extension(
        name="trajectoryclusteringanalysis.optimal_matching",
        sources=["src/trajectoryclusteringanalysis/optimal_matching.c"],
        include_dirs=[np.get_include()],
        #language="c",  # Compilation en C pour plus de rapidité
        extra_compile_args=compile_args,
        extra_link_args=link_args
    )
]

setup(
    name='trajectoryclusteringanalysis',
    version=get_version(),
    author='Nicolas and Ndiaga',
    author_email='ndiagadiengs1@gmail.com',
    description='Un package pour l’analyse des trajectoires de soins par clustering',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  
    install_requires=get_requirements('requirements.txt'),
    #ext_modules=cythonize(extensions,  compiler_directives={'boundscheck': False, 'wraparound': False, 'cdivision': True, 'language_level': 3}),
    ext_modules=extensions, 
    include_package_data=True,  # Active MANIFEST.in
    classifiers=[
        'Programming Language :: Python :: 3',
        #'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    python_requires='>=3.6',
)
