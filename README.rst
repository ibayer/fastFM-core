.. image:: https://travis-ci.org/ibayer/fastFM-core.svg
    :target: https://travis-ci.org/ibayer/fastFM-core

GIT CLONE INSTRUCTION
=====================
This repository requires sub-repositories and just using ``git clone ..``
**doesn't fetch** them. Use
``git clone --recursive ..``
instead.

Otherwise you have to run ``git submodule update --init --recursive`` **from within** the
``fastFM-core/`` folder in order to get the sub-repositories.


DEPENDENCIES
============
* CXSparse (included)
* gsl 1.15-1 (only testsuite)
* glib-2.0 (only testsuite)
* argp (only cli, included by default in Linux)

install depenencies:
-------------------
this worked on ubuntu 14.04:
``sudo apt-get install libglib2.0-dev libatlas-base-dev``
and for the testsuite also ``sudo apt-get install libgsl0-dev``

Install on OSX
===============
Library compiles on OSX, however console interface doesn't.

Recommended way to manage dependencies is `Homebrew package manager
<https://brew.sh>`_. If you have brew installed, dependencies can be installed by running command ``brew install glib argp-standalone``.

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
* check coding style (google) ``lang-format-3.5 -style=google -i <YOUR_FILE.c/h>``
* static code analysis ``clang-tidy-3.5 -fix  <YOUR_FILE.c/h> -- I.``

cli example:

./fastFM-core/bin/fastfm data/train_regression data/test_regression --task regression --init-var=0.11 --n-iter=123 --solver='mcmc' --rank 7 --l2-reg=.22
