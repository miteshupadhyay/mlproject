from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT= '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''

    requiremts = []
    with open(file_path) as file_obj:
        requiremts = file_obj.readlines()
        requiremts = [req.replace("\n","") for req in requiremts]

        if HYPEN_E_DOT in requiremts:
            requiremts.remove(HYPEN_E_DOT)

setup(
    name= 'mlproject',
    version='1.0.0',
    author='mitesh',
    author_email='upadhyaymitesh91@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)