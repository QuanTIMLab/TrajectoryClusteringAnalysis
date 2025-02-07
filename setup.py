from setuptools import find_packages, setup
from typing import List

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

setup(
    name='TrajectoryClusteringAnalysis',
    version='0.0.1',
    author='Nicolas and Ndiaga',
    author_email='ndiagadiengs1@gmail.com',
    description='Un package pour l’analyse des trajectoires de soins par clustering',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
