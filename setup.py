from setuptools import find_packages, setup

setup(name="easy-o3d",
      version="1.0",
      description="An easy-to-use wrapper around some of Open3D's registration functionality.",
      long_description=open("README.md").read(),
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      author="Matthias Humt",
      author_email="matthias dot humt at mailbox dot org",
      python_requires=">=3.6.0",
      url="https://github.com/hummat/easy-o3d",
      install_requires=["open3d>=0.14.1", "tqdm>=4.62.3", "tabulate>=0.8.9"],
      extras_require={"scripts": ["scikit-optimize>=0.8.1", "matplotlib>=3.3"],
                      "test": ["pytest>=6.2.3", "plotly>=4.14.3"]},
      include_package_data=True,
      license='GPLv3',
      entry_points={"console_scripts": ["run = scripts.run_registration:main",
                                        "hyperopt = scripts.hyperopt:main"]})
