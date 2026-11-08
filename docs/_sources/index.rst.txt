GLIESE486b Documentation
========================

GLIESE486b uses the Daneel Python package to model exoplanet transits.
The command-line tool can plot a synthetic light curve for GJ 486 b or
compare light curves from multiple YAML configuration files.

Source code and examples are available in the
`GLIESE486b GitHub repository <https://github.com/chiara-guglielmo/GLIESE486b>`_.

Installation
------------

Use Python 3.10 and install Daneel from the repository.

.. code-block:: bash

   git clone https://github.com/chiara-guglielmo/GLIESE486b.git
   cd GLIESE486b
   python -m pip install .

Usage
-----

Plot the transit light curve for GJ 486 b using the default parameters.

.. code-block:: bash

   daneel --transit

Use ``--input`` to provide one or more YAML files, each with a non-empty
``transit`` mapping. The following command compares two configurations and
saves a PNG in the current directory.

.. code-block:: bash

   daneel --transit --input examples/G486b.yaml examples/G486b_halfR.yaml --output

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
