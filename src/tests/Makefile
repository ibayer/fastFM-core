P= test_ffm_als_mcmc test_ffm_utils test_ffm_sgd test_random
VPATH = ../
OBJECTS= ffm_als_mcmc.o ffm_sgd.o
INCLUDES = -I.. 
CFLAGS = `pkg-config --cflags gsl glib-2.0` -std=c99 -g -Wall -O0 $(INCLUDES)
LDLIBS= `pkg-config --libs gsl glib-2.0` -L../../externals/CXSparse/Lib -lcxsparse

CC=gcc

all: test_ffm_sgd test_ffm_als_mcmc test_ffm_utils test_random
	( cd ../../externals/CXSparse ; $(MAKE) library )
	( export USE_THREAD=0; cd externals/OpenBLAS ; $(MAKE) libs)

check: 
	gtester test_random --keep-going --verbose
	gtester test_ffm_utils --keep-going --verbose
	gtester test_ffm_sgd --keep-going --verbose
	gtester test_ffm_als_mcmc --keep-going --verbose

mem_check:
	G_DEBUG=gc-friendly G_SLICE=always-malloc valgrind --leak-check=full ./test_ffm_als_mcmc

test_ffm_sgd : kmath.o ffm_random.o ffm_utils.o ffm_als_mcmc.o ffm_sgd.o TestFixtures.o
test_ffm_als_mcmc : kmath.o ffm_random.o ffm_utils.o ffm_als_mcmc.o TestFixtures.o
test_ffm_utils : kmath.o ffm_random.o ffm_utils.o ffm_als_mcmc.o
test_random: kmath.o ffm_random.o ffm_utils.o

kmath.o: fast_fm.h
ffm_random: fast_fm.h
test_ffm_sgd.o: fast_fm.h
ffm_als_mcmc.o : fast_fm.h
ffm_utils.o : fast_fm.h
TestFixtures.o : TestFixtures.h


.PHONY : clean
clean : 
	rm -f $(P) $(OBJECTS)
	rm -f *.o
