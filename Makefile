
lib: $(OBJECTS)
	( cd src ; $(MAKE) lib )

cli:
	( cd src ; $(MAKE) cli )

.PHONY : clean
clean :
	( cd src ; $(MAKE) clean )
