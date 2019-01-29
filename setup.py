import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dw",
    version="0.0.3",
    author="Dennis Wittich",
    author_email="denniswittich@hotmail.de",
    description="Mixed tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
)