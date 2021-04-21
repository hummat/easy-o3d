from setuptools import find_packages, setup

setup(name="easy-o3d",
      version="0.1",
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      description="Uses Open3D functionality to implement various 3D data based algorithm used at RMC",
      long_description=open("README.md").read())
