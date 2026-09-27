import os
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ur_simulation",
    description="Repository for Classic and Reinforcement learning based control of the UR7e",
    author="Robin Moret",
    author_email="r.l.moret@student.rug.nl",
    packages=["ur_simulation"],
    include_package_data=True,
    package_data={
        "ur_simulation": ["version.txt", "assets/**/*"],
    },
    version="0.0.1",
    install_requires=["pybullet", "numpy<2"],
)

