from setuptools import setup, find_packages

setup(
    name="ml-field-experiments",
    version="0.0.0",
    description="",
    url="https://github.com/blei-lab/ml-field-experiments",
    author="blei-lab",
    author_email="adj2147@columbia.edu",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "openpyxl",
        "pandas",
        "PyYAML",
        "seaborn",
        "scikit-learn",
        "statsmodels",
        "lxml",
        "econml",
        "cvxopt",
    ],
    entry_points={},
)
