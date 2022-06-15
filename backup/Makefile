# variables
FC=gfortran
CFLAGS= -g -Og

objs=main.f90 modglobal.f90
stamp: 	$(objs)
		$(FC)  $(CFLAGS) $(objs) -o a.out

# compiling
main.o: main.f90
	$(FC) -c $(CFLAGS) main.f90
modglobal.o: modglobal.f90
	$(FC) -c $(CFLAGS) modglobal.f90

#cleanup
clean:
	rm *.o a.out

#run
run:
	make
	./a.out
