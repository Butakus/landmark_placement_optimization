from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='landmark_placement_optimization',
    version='0.0.1',
    description='Landmark Placement Optimization package',
    long_description=readme,
    author='Francisco Miguel Moreno',
    author_email='franmore@ing.uc3m.es',
    url='https://github.com/butakus/landmark_placement_optimization',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
