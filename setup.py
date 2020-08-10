import re
from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution

with open("README.md", "r") as fh:
    long_description = fh.read()
INSTALL_REQUIRES = [
    "rasterio == 1.1.5",
    "affine == 2.3.0",
    "numpy == 1.19.1",
]

# If first not installed install second package
CHOOSE_INSTALL_REQUIRES = [("opencv-python>=4.1.1", "opencv-python-headless>=4.1.1")]


def choose_requirement(main, secondary):
    """If some version version of main requirement installed, return main,
    else return secondary.
    """
    try:
        name = re.split(r"[!<>=]", main)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary
    return str(main)


def get_install_requirements(install_requires, choose_install_requires):
    for main, secondary in choose_install_requires:
        install_requires.append(choose_requirement(main, secondary))
    return install_requires


setup(
    name="stitch_n_split",
    version="0.0.4",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    url="https://github.com/cypherics/ShapeMerge",
    description="Library for stitching and spliting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires="~=3.3",
    install_requires=get_install_requirements(
        INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES
    ),
    keywords=["GIS, Rasterio, Sticth, Split, Mesh, Grid, Geo Reference"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
    ],
)
