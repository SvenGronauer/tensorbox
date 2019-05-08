import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorbox",
    version="0.0.1",
    author="Sven Gronauer",
    author_email="sven.gronauer@tum.de",
    description="A package for machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sven.gronauer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)