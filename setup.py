from setuptools import find_packages, setup
from typing import List
import sys

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) ->List[str]:
    '''
    Return the list of requirements
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

sys.path.append('D:/Study/Education/Projects/ML1_midterm/main')

setup(
    name = 'mlproject1',
    version = '0.0.1',
    author = 'PhamDuyAnh',
    author_email = 'phamduyanh0816@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirement.txt')
)