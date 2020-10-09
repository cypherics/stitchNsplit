from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
install_requires = [
    "rasterio == 1.1.5",
    "opencv-python >= 4.1.1",
    "affine == 2.3.0",
    "numpy == 1.19.1",
    "image-fragment==0.2.2",
]

setup(
    name="stitch_n_split",
    version="0.1.0",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    description="Library for stitching and spliting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires="~=3.3",
    install_requires=install_requires,
    keywords="GIS Rasterio Sticth Split Mesh Grid Geo Reference",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)