.. mobilkit documentation master file, created by
   sphinx-quickstart on Thu Aug 13 13:33:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mobilkit's documentation!
====================================

`mobilkit` is a library for analyzing human mobility data in Python, leveraging on the `Dask` framework for faster, parallel computation.

 The library is in continuos development and currently allows to:

* Load and filter raw mobility data covering large spatial and temporal extensions.

* Compute the user statistics (number of active days, number of positions recorded) and filter them accordingly to the analysis.

* extract home and work locations of users based on a given tessellation.

* compute the land use of a given urban region

* characterize the displacement of people under different grouping (distance from an epicenter, socio-economic index, etc.) after a major event

Documentation
=============

Besides this documentation, many example notebooks can be found in the original repo under the `docs/examples <https://github.com/mindearth/mobilkit/tree/main/docs/examples>`_ folder.

Detailed notebooks with all the functionalities shown are found in `examples/ <https://github.com/mindearth/mobilkit/tree/main/examples>`_.


Collaborate with us
=====================

`mobilkit` is an active project and any contribution is welcome.

You are encouraged to report any issue or problem encountered while using the software or to seek for support.

If you would like to contribute or add functionalities to `mobilkit`, feel free to fork the project, open an issue and contact us.

Installation
============

.. note::
   `mobilkit` will install a complete installation of Dask, so consider installing it in a virtualenv to connect to an existing dask cluster.

.. note::
   You can try `mobilkit` without installing it on Binder, just click below
.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/mindearth/mobilkit/main?filepath=docs%2Fexamples%2Fmobilkit_tutorial.ipynb

#. Create an environment `mobilkit`

    .. code-block:: console

      python3 -m venv mobilkit

#. Activate

    .. code-block:: console

      source mobilkit/bin/activate

#. Update pip to latest version in the environment

    .. code-block:: console

      pip install --upgrade pip

#. Install mobilkit

    .. code-block:: console

      pip install mobilkit

#. OPTIONAL to use `mobilkit` on the jupyter notebook

   - Activate the virutalenv:

        .. code-block:: console

          source mobilkit/bin/activate

   - Install jupyter notebook:

        .. code-block:: console

          pip install jupyter

   - Run jupyter notebook

        .. code-block:: console

          jupyter notebook

   - (Optional) install the kernel with a specific name

        .. code-block:: console

          ipython kernel install --user --name=mobilkit_env


If you already have `scikit-mobility <https://github.com/scikit-mobility/scikit-mobility>`_ installed, skip the environment creation and run these commands from the `skmob` anaconda environment.

`mobilkit` by default will only install core packages needed to run the main functions. There are three optional packages of dipendencies (the `mobilkit[complete]` installs everything):

* `[viz]` will install `contextily`, needed to visualize map backgrounds in certain viz functions;

* `[doc]` will install all the needed packages to build the docs;

* `[skmob]` will install `scikit-mobility` as well.

Test the installation
=====================

.. code-block:: console

  > source activate mobilkit
  (mobilkit) > python
  >>> import mobilkit

Citing
======

If you use `mobilkit` please cite us: 

.. note::
  Enrico Ubaldi, Takahiro Yabe, Nicholas K. W. Jones, Maham Faisal Khan, Satish V. Ukkusuri, Riccardo Di Clemente and Emanuele Strano
  Mobilkit: A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics using High Frequency Human Mobility Data,
  2021, KDD 2021 Humanitarian Mapping Workshop, https://arxiv.org/abs/2107.14297

Bibtex:
    @misc{ubaldi2021mobilkit,
        title={Mobilkit: A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics using High Frequency Human Mobility Data},
        author={Enrico Ubaldi and Takahiro Yabe and Nicholas K. W. Jones and Maham Faisal Khan and Satish V. Ukkusuri and Riccardo Di Clemente and Emanuele Strano},
        year={2021},
        eprint={2107.14297},
        primaryClass={cs.CY},
        archivePrefix={arXiv}}

      
    
  

Credits and contacts
====================

This code has been developed by `Mindearth <https://mindearth.ch>`_, the `Global Facility for Disaster Reduction and Recovery <https://www.gfdrr.org/en>`_ (GFDRR) and `Purdue University <https://www.purdue.edu/>`_.

Funding was provided by the Spanish Fund for Latin America and the Caribbean (SFLAC) under the Disruptive Technologies for Development (DT4D) program.

The findings, interpretations, and conclusions expressed in this repository and in the example notebooks are entirely those of the authors. They do not necessarily represent the views of the International Bank for Reconstruction and Development/World Bank and its affiliated organizations, or those of the Executive Directors of the World Bank or the governments they represent.

The code is released under the MIT license (see the LICENSE file for details).


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples/mobilkit_tutorial.ipynb
   examples/M4R_01_Dataloading_cleaning_validating.ipynb
   examples/M4R_02_DisplacementAnalysis.ipynb
   examples/M4R_03_POI_visit_analysis.ipynb
   examples/M4R_04_Population_Density_Analysis.ipynb
   examples/USS01_Mumbai.ipynb
   examples/USS02_CityComparison.ipynb
   reference/Loading_data.rst
   mobilkit

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


