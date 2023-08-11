![GitHub release (latest by date)](https://img.shields.io/github/v/release/mindearth/mobilkit)
![GitHub](https://img.shields.io/github/license/mindearth/mobilkit)
![GitHub contributors](https://img.shields.io/github/contributors/mindearth/mobilkit)
[![Documentation Status](https://readthedocs.org/projects/mobilkit/badge/?version=latest)](https://mobilkit.readthedocs.io/en/latest/?badge=latest)


# mobilkit

A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics using High Frequency Human Mobility Data.

`mobilkit` provides a set of tools to analyze mobility traces to assess the users response to extreme events.
Try `mobilkit` without installing it in a MyBinder notebook:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mindearth/mobilkit/main?filepath=docs%2Fexamples%2Fmobilkit_tutorial.ipynb)

## Table of contents
1. [Documentation](#documentation)
1. [Collaborate with us](#collaborate)
1. [Installation](#installation)
	- [with pip](#installation_pip)
	- [with conda](#installation_conda)
	- [test installation](#test_installation)
1. [Tutorials](#tutorials)
1. [Examples](#examples)
	- [Quickstart](#quickstart)
1. [Citing](#citing)
1. [Credits and contacts](#credits)
    
<a id='documentation'></a>
## Documentation

Full documentation with examples can be found online [here](https://mobilkit.readthedocs.io/en/latest/), otherwise see the notebooks in [docs/examples](docs/examples/) for a step-by-step coverage of the library or the ones in [examples/](examples/) for a more detailed showcase of the package's capabilities.


<a id='collaborate'></a>
## Collaborate with us
`mobilkit` is an active project and any contribution is welcome.

You are encouraged to report any issue or problem encountered while using the software or to seek for support.

If you would like to contribute or add functionalities to `mobilkit`, feel free to fork the project, open an issue and contact us.

<a id='installation'></a>
## Installation

<a id='installation_pip'></a>    
### Install with pip

Start by creating an environment and install mobilkit there.

1. Create an environment `mobilkit`

        python3 -m venv mobilkit
		# or, on Windows
		python -m venv c:\path\to\mobilkit

2. Activate
    
        source mobilkit/bin/activate
		# or, on Windows
		c:\path\to\mobilkit\Scripts\activate.bat

3. Update pip 

        pip install --upgrade pip

4. Install `mobilkit` (this will also install `Dask` and all the needed modules)

        pip install mobilkit


5. OPTIONAL to use `mobilkit` on the jupyter notebook

	- Activate the virutalenv:
	
			source mobilkit/bin/activate
	
	- Install jupyter notebook:
		
			pip install jupyter 
	
	- Run jupyter notebook
			
			jupyter notebook
			
	- (Optional) install the kernel with a specific name to your existing notebook server
			
			source mobilkit/bin/activate
			pip install ipykernel
			ipython kernel install --user --name=mobilkit_env
		

If you already have [`scikit-mobility`](https://github.com/scikit-mobility/scikit-mobility) installed, skip the environment creation and run these commands from the skmob anaconda environment.

`mobilkit` by default will only install core packages needed to run the main functions. There are three optional packages of dipendencies (the `mobilkit[complete]` installs everything):
- `[viz]` will install `contextily`, needed to visualize map backgrounds in certain viz functions;
- `[doc]` will install all the needed packages to build the docs;
- `[skmob]` will install `scikit-mobility` as well;
- `[locations]` will also install [`infostop`](https://github.com/ulfaslak/infostop) to detect users' typical locations.

<a id='installation_conda'></a>
### Install with conda
**TODO**

<a id='test_installation'></a>
### Test the installation

```
> source activate mobilkit
(mobilkit)> python
>>> import mobilkit
>>>
```
<a id='examples'></a>
## Examples

Several notebooks are found in the [docs/examples](docs/examples/) folder, we resume here the most important ones.

<a id='quickstart'></a>
### Quickstart
We show the basic usage and functionalities in the [mobilkit_tutorial.ipynb](docs/examples/mobilkit_tutorial.ipynb) notebook.

<a id='citing'></a>
## Citing
If you use `mobilkit` please cite us: 

> Enrico Ubaldi, Takahiro Yabe, Nicholas K. W. Jones, Maham Faisal Khan, Satish V. Ukkusuri, Riccardo Di Clemente and Emanuele Strano
> **Mobilkit: A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics using High Frequency Human Mobility Data**,
> 2021, KDD 2021 Humanitarian Mapping Workshop, https://arxiv.org/abs/2107.14297

Bibtex:
```
@misc{ubaldi2021mobilkit,
    title={Mobilkit: A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics using High Frequency Human Mobility Data},
    author={Enrico Ubaldi and Takahiro Yabe and Nicholas K. W. Jones and Maham Faisal Khan and Satish V. Ukkusuri and Riccardo {Di Clemente} and Emanuele Strano},
    year={2021},
    eprint={2107.14297},
    primaryClass={cs.CY},
    archivePrefix={arXiv},
}
```

<a id='credits'></a>
## Credits and contacts
This code has been developed by [Mindearth](https://mindearth.ch), the [Global Facility for Disaster Reduction and Recovery](https://www.gfdrr.org/en) (GFDRR) and [Purdue University](https://www.purdue.edu/).

Funding was provided by the Spanish Fund for Latin America and the Caribbean (SFLAC) under the Disruptive Technologies for Development (DT4D) program.

The code is released under the MIT license (see the LICENSE file for details).
