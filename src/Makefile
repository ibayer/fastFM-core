OBJECTS= kmath.o ffm_random.o ffm_als_mcmc.o ffm_utils.o ffm_sgd.o ffm.o
CFLAGS = -std=c99 -fPIC -g -Wall -O3 -I/include -I./src -I../externals/CXSparse/Include
LDLIBS= -L../externals/CXSparse/Lib -lcxsparse -lcblas -lm
CC=gcc

ifeq ($(shell uname -s),Darwin)
	CFLAGS += -I$(shell find /System/Library/Frameworks/Accelerate.framework -name 'cblas.h' | xargs dirname)
endif

$(P): CFLAGS += -DTest_operations 
$(P): $(OBJECTS)

ffm_als_mcmc.o : fast_fm.h
ffm_utils.o : fast_fm.h
ffm_sgd.o : fast_fm.h
ffm.o : fast_fm.h
cli.o : fast_fm.h


lib: $(OBJECTS)
	( cd ../externals/CXSparse ; $(MAKE) library )
	mkdir -p ../bin/
	ar rcs ../bin/libfastfm.a $(OBJECTS)

cli: $(OBJECTS)
	( cd ../externals/CXSparse ; $(MAKE) library )
	mkdir -p ../bin/
	$(CC) $(OBJECTS) cli.o $(CFLAGS) $(LDLIBS) -o ../bin/fastfm

.PHONY : clean
clean :
	rm -f $(P) $(OBJECTS)
	rm -f ../bin/libfastfm.a
	rm -f ../bin/fastfm
