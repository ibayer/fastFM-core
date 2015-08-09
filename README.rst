A short paper describing the library is now available on 
arXiv http://arxiv.org/abs/1505.00641

GIT CLONE INSTRUCTION
=====================
This repository relays on sub-repositories just using ``git clone ..``
**doesn't fetch** them.

``git clone --recursive ..``

Or do the two-step dance if you wish.
You need to run ``git submodule update --init --recursive`` **from within** the
``fastFM-core/`` folder in order to clone them as well.


DEPENDENCIES
============
* glib-2.0
* CXSparse (included)
* gsl 1.15-1 (only testsuite)
* argp (included by default in Linux)

install depenencies:
-------------------
this worked on ubuntu 14.04:
``sudo apt-get install libglib2.0-dev libatlas-base-dev``
and for the testsuite also ``sudo apt-get install libgsl0-dev``

Install on OSX
===============
Library compiles on OSX, however console interface doesn't.

Recommended way to manage dependencies is `Homebrew package manager
<https://brew.sh>`_. If you have brew installed, dependencies can be installed by running command ``brew install glib gsl argp-standalone``.

Install on Windows
========================
It should be possible to compile the library on Windows.
I'm developing on linux but have received multiple requests from people who
want to run this library on other platforms.
Please let me know about issues you ran into or how you manged to compile on
other platfroms (or just open a PR) so that we include this information here.

compile command line linterface
===============================
compile cli: ``make cli``
you can find the executible in ``fastFM-core/bin``.


demo
====
``demo/data/`` contains sample input files for various tasks
and examples how to use fastFM's CLI interface can be found in
``demo/Makefile``

how to run tests
----------------

* cd ``src/tests/``
* bulid tests ``make all``
* run all tests ``make check``
* run valgrind memory check on sparse_test.c ``make mem_check``
* run valgrind to check for errors ``valgrind -v ./a.out >& out``

cli example:

./fastFM-core/bin/fastfm data/train_regression data/test_regression --task regression --init-var=0.11 --n-iter=123 --solver='mcmc' --rank 7 --l2-reg=.22
