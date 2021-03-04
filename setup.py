from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pywave",
    version="0.1",
    description="A hybrid Python/C wave propagator",
    long_description="",
    author="Jaime Freire de Souza",
    author_email="jaimefreire.souza@gmail.com",
    license="BSD-2-Clause License",
    packages=find_packages(),
    install_requires=required,
)
