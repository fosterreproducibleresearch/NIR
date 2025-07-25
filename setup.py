import re
from setuptools import setup, find_packages

# Min version :pip3 install -e .
# Dev version :pip3 install -e .["dev"]

_deps = [
    "torch>=2.2.0",
    "lightning>=2.1.3",
    "pandas>=2.1.0",
    "numpy==1.26.4",
    "polars==1.9.0",
    "scikit-learn>=1.2.2",
    "pyarrow>=11.0.0",
    "pykeen>=1.10.2",
    "zstandard>=0.21.0",
    "pytest>=7.2.2",
    "psutil>=5.9.4",
    "ruff>=0.0.284",
    "gradio>=3.23.0",
    "rdflib>=7.0.0",
    "tiktoken>=0.5.1",
    "dicee>=0.1.4",
    "ontolearn>=0.9.2",
    "matplotlib>=3.8.2"
]

# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = dict()
extras["min"] = deps_list(
    "pandas",
    #"ontolearn",
    "polars", "pyarrow", "rdflib",  # Loading KG
    "torch", "lightning",  # Training KGE
    "tiktoken",  # used for BPE
    "matplotlib",  # Unclear why it is needed
    "numpy"
)

# TODO: Remove polars, rdflib, tiktoken, psutil, matplotlib from min

extras["dev"] = (extras["min"] + deps_list("ruff", "pytest",
                                           "polars", "pyarrow",
                                           "scikit-learn"))
install_requires = [extras["min"]]

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name="nir",
    description="Compositional Neural Instance Retriever",
    version="0.1.0",
    packages=find_packages(),
    extras_require=extras,
    install_requires=list(install_requires),
    author="N\'Dah Jean Kouagou",
    author_email='jeank@aims.ac.za',
    url='https://github.com/dice-group/CoNeuralReasoner',
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License"],
    python_requires='>=3.10',
    entry_points={"console_scripts":
                      ["nir=nir.scripts.run:main",
                       "nirindex=nir.scripts.index:main",
                       "nirserve=nir.scripts.serve:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)