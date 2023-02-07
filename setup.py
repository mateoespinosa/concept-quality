from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

version = '0.1.0'
setup(
    name='concepts_xai',
    version=version,
    packages=find_packages(),
    description='Concept Extraction Comparison',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
)
