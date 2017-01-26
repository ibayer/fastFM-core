
lib: $(OBJECTS)
	( cd externals/CXSparse ; $(MAKE) library )
	( export USE_THREAD=0; cd externals/OpenBLAS ; $(MAKE) libs)
	( cd src ; $(MAKE) lib )

cli:
	( cd externals/CXSparse ; $(MAKE) library )
	( export USE_THREAD=0; cd externals/OpenBLAS ; $(MAKE) libs)
	( cd src ; $(MAKE) cli )

.PHONY : clean
clean :
	( cd src ; $(MAKE) clean )

.PHONY : clean_libs
clean_libs :
	( cd externals/OpenBLAS/ ; $(MAKE) clean )
	( cd externals/CXSparse/ ; $(MAKE) clean )
