OBJECTS = kmath.o ffm_random.o ffm_als_mcmc.o ffm_utils.o ffm_sgd.o ffm.o
INCLUDES =  -I../externals/OpenBLAS
LDLIBS = -L../externals/CXSparse/Lib -lcxsparse  -L../externals/OpenBLAS -lopenblas -lm
CFLAGS = -std=c99 -fPIC -g -Wall -O3 $(INCLUDES)
CC=gcc

$(P): CFLAGS += -DTest_operations 
$(P): $(OBJECTS)

ffm_als_mcmc.o : fast_fm.h
ffm_utils.o : fast_fm.h
ffm_sgd.o : fast_fm.h
ffm.o : fast_fm.h
cli.o : fast_fm.h


lib: $(OBJECTS)
	mkdir -p ../bin/
	ar -x ../externals/CXSparse/Lib/libcxsparse.a
	ar -x ../externals/OpenBLAS/libopenblas.a
	ar rcs ../bin/libfastfm.a $(OBJECTS) *.o

cli: $(OBJECTS) cli.o
	mkdir -p ../bin/
	$(CC) $(OBJECTS) cli.o $(CFLAGS) $(INCLUDES) $(LDLIBS) -o ../bin/fastfm

.PHONY : clean
clean :
	rm -f $(P) $(OBJECTS)
	rm -f  *.o
	rm -f ../bin/libfastfm.a
	rm -f ../bin/fastfm
