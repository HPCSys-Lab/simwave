import versioneer
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md") as f:
    readme = f.read()

setup(
    name="simwave",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Finite difference 2D/3D acoustic wave propagator.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/HPCSys-Lab/simwave',
    author="HPCSys-Lab",
    author_email="senger.hermes@gmail.com",
    license="GPL-3.0 License",
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ]
)
