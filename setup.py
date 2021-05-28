import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="advpipe-kubic71",
    version="0.0.1",
    author="Jakub Hejhal",
    author_email="kubic71@gmail.com",
    description="Black-box Adversarial attack pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kubic71/bachelors-thesis",
    project_urls={
        "Bug Tracker": "https://github.com/kubic71/bachelors-thesis/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
