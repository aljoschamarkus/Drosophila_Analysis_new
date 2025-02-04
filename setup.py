from setuptools import setup, find_packages

setup(
    name="package",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "opencv-python",
    ],
)