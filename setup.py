import versioneer
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="simwave",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Finite difference 2D/3D acoustic wave propagator.",
    long_description="""Simwave is an hybrid Python/C tool that
    simulates the propagation of the acoustic wave using the
    finite difference method in 2D and 3D domains.""",
    url='https://github.com/HPCSys-Lab/simwave',
    download_url=r'''
    https://github.com/HPCSys-Lab/simwave/archive/refs/tags/v0.5.tar.gz
    ''',
    author="HPCSys-Lab",
    author_email="senger.hermes@gmail.com",
    license="GPL-3.0 License",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Development Status :: 4 - Beta'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Simulation',
        'License :: OSI Approved :: GPL-3.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
