import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mdsim",
    version="0.0.1",
    author="Cameron Perot",
    description="A basic molecular dynamics simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cameronperot/",
    packages=["mdsim"],
    package_dir={"mdsim": "src/mdsim"},
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.6",
    install_requires=["numpy", "pandas", "matplotlib", "imageio"],
)
