from setuptools import setup, find_packages

long_description = open('README.md').read()

with open('requirements.txt') as f:
    REQUIRED_PKGS = f.read().splitlines()

TESTS_REQUIRES = ['pytest']
EXTRAS_REQUIRE = {'test': TESTS_REQUIRES}

setup(
    name='mobilkit',
    version='0.1.0',
    extras_require=EXTRAS_REQUIRE,
    license='MIT',
    python_requires='>=3.6',
    description='A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics using High Frequency Human Mobility Data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer='MindEarth',
    maintainer_email='enrico.ubaldi@mindearth.org',
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 ],
    install_requires=REQUIRED_PKGS,
    author="MindEarth",
    author_email='enrico.ubaldi@mindearth.org',
    url="https://github.com/mindearth/mobilkit",
    project_urls={
        "Bug Tracker": "https://github.com/mindearth/mobilkit/issues",
    },
    packages=find_packages(include=["mobilkit","mobilkit.*"], exclude=["data","results","examples"]),
)
