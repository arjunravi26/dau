from setuptools import setup, find_packages

setup(
    name="dau-undersampling",
    version="0.1.0",
    author="Arjun Ravi",
    author_email="arjunravi726@gmail.com",
    description="Density Aware Undersampling for Imbalanced Data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arjunravi26/dau-undersampling",  # your repo
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
