If you use this project please give credit by citing:

    Immanuel Bayer (2014): fastFM: A Library for Factorization Machines http://arxiv.org/abs/1505.00641

.. image:: https://travis-ci.org/ibayer/fastFM-core.svg
    :target: https://travis-ci.org/ibayer/fastFM-core

.. image:: https://img.shields.io/badge/platform-OSX|Linux-lightgrey.svg
    :target: https://travis-ci.org/ibayer/fastFM

.. image:: https://img.shields.io/pypi/l/Django.svg   
    :target: https://travis-ci.org/ibayer/fastFM

fastFM: A Library for Factorization Machines
============================================

This repository contains the source code for the fastFM C library and the stand-alone
comand line interface (cli). In general we recommend to use fastFM through the high level `Python
interface <https://github.com/ibayer/fastFM>`_.

Usage
-----


.. code-block:: bash

    fastFM-core/bin/bin/fastfm data/train_regression data/test_regression \
	--task regression   \
	--rng-seed 1234     \
	--init-var=0.11     \
	--n-iter=123        \
	--solver='mcmc'     \
	--rank 7            \
	--l2-reg=.22


Examples on how to use the other command line options options, including example data, can be found
in ``fastFM-core/demo/Makefile``. The ``demo/`` folder contains examples showing how to use
fastFM as C library.

+----------------+------------------+-----------------------------+
| Task           | Solver           | Loss                        |
+================+==================+=============================+
| Regression     | als, mcmc, sgd   | Square Loss                 |
+----------------+------------------+-----------------------------+
| Classification | als, mcmc, sgd   | Probit(Map), Probit, Sigmoid|
+----------------+------------------+-----------------------------+
| Ranking        | sgd              | BPR                         |
+----------------+------------------+-----------------------------+
*Supported solvers and tasks*

Installation
------------

**OS X:**
Library compiles on OSX, however console interface doesn't.

.. code-block::

    # Install cblas (Linux only).
    $ sudo apt-get libatlas-base-dev

    # Clone the repro including submodules (or clone + `git submodule update --init --recursive`)
    $ git clone --recursive https://github.com/ibayer/fastFM-core.git

    # Build library
    $ cd fastFM-core/; make;

    # Build command line interface (this works currently only on osx)
    $ make cli

Tests
-----

**OS X:**
Recommended way to manage dependencies is `Homebrew package manager <https://brew.sh>`_.
If you have brew installed, dependencies can be installed by running command
``brew install glib gsl argp-standalone``.

.. code-block::

    # The tests require the glib and gsl library (Linux, for OSX see above).
    $ sudo apt-get libglib2.0-dev libgsl0-dev

    $ cd fastFM-core/src/tests

    # Build the tests
    $ make

    # Run all tests
    $ make check


Contribution
------------

* Star this repository: keeps contributors motivated
* Open a issue: report bugs or suggest improvements
* Fix errors in the documentation: small changes matter
* Contribute code

**Contributions are very wellcome!** Since this project lives on github we reommend
to open a pull request (PR) for code contributions as early as possible. This is the
fastest way to get feedback and allows `Travis CI <https://travis-ci.org/ibayer/fastFM-core>`_ to run checks on your changes.

**Development Guidlines**

* check coding style (google) ``lang-format-3.5 -style=google -i <YOUR_FILE.c/h>``
* static code analysis ``clang-tidy-3.5 -fix  <YOUR_FILE.c/h> -- I.``
* run valgrind memory check on sparse_test.c ``make mem_check``
* run valgrind to check for errors ``valgrind -v ./a.out >& out``


**Contributors**

* takuti
* altimin
* ibayer

License: BSD
------------
