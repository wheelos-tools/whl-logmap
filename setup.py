import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="whl-logmap",
    version="0.0.1",
    author="daohu527",
    author_email="daohu527@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wheelos-tools/whl-logmap",
    project_urls={
        "Bug Tracker": "https://github.com/wheelos-tools/whl-logmap/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    entry_points={
        'console_scripts': [
            'whl_logmap = whl_logmap.main:main',
        ],
    },
    python_requires=">=3.6",
)
