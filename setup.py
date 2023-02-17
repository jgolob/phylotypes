import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phylotypes",
    version="0.0.6",
    author="Jonathan Louis Golob",
    author_email="j-dev@golob.org",
    description="Group phylogenetically placed sequence variants into phylotypes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgolob/phylotypes",
    project_urls={
        "Bug Tracker": "https://github.com/jgolob/phylotypes/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'biopython',
        'scikit-bio',
        'numpy',
        'scikit-learn',
        'taichi',
    ],
    entry_points={
        'console_scripts': [
            'phylotypes=phylotypes.phylotypes:main',
            'add_phylotypes=phylotypes.add_phylotypes:main',
        ],
    },
)
