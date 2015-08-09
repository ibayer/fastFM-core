OBJECTS= kmath.o ffm_random.o ffm_als_mcmc.o ffm_utils.o ffm_sgd.o ffm.o  cli.o
CFLAGS = `pkg-config --cflags glib-2.0` -std=c99 -fPIC -g -Wall -O3 -I/include -I./src -I../externals/CXSparse/Include
LDLIBS= `pkg-config --libs glib-2.0` -L../externals/CXSparse/Lib -lcxsparse -lcblas -latlas -lm
CC=gcc


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
	$(CC) $(OBJECTS) $(CFLAGS) $(LDLIBS) -o ../bin/fastfm

.PHONY : clean
clean :
	rm -f $(P) $(OBJECTS)
	rm -f ../bin/libfastfm.a
	rm -f ../bin/fastfm
