from setuptools import find_packages, setup

setup(name="easy-o3d",
      version="1.0",
      description="An easy-to-use wrapper around some of Open3D's registration functionality.",
      long_description=open("README.md").read(),
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      author="Matthias Humt",
      author_email="matthias dot humt at mailbox dot org",
      python_requires=">=3.10.10",
      url="https://github.com/hummat/easy-o3d",
      install_requires=["open3d>=0.17.0", "tabulate>=0.9.0"],
      extras_require={"scripts": ["scikit-optimize>=0.9.0", "matplotlib>=3.7.1"],
                      "test": ["pytest>=7.2.2", "plotly>=5.13.1"]},
      include_package_data=True,
      license='GPLv3',
      entry_points={"console_scripts": ["run = scripts.run_registration:main",
                                        "hyperopt = scripts.hyperopt:main"]})
