CC = icc

#LIBS = -L/home/foureste/Projects/OpenBLAS-0.2.14/ -lopenblas 
LIBS = -mkl
INCLUDES =
CFLAGS = -Ofast -qopt-report=2 -qopt-report-phase=vec -xCORE-AVX512 -qopt-zmm-usage=high -qopenmp -g -I./

all:    dgemm-simple dgemm-splat zgemm-splat python-tester

dgemm-splat: dgemm.o dgemm-splat.o dgemm-splat.so
	$(CC) -o $@ dgemm.o dgemm-splat.o $(CFLAGS) $(LIBS) $(INCLUDES)

dgemm-simple: dgemm.o dgemm-simple.o dgemm-simple.so
	$(CC) -o $@ dgemm.o dgemm-simple.o $(CFLAGS) $(LIBS) $(INCLUDES)

zgemm-splat: zgemm.o zgemm-splat.o zgemm-splat.so
	$(CC) -o $@ zgemm.o zgemm-splat.o $(CFLAGS) $(LIBS) $(INCLUDES)

python-tester: python-tester.o python-tester.so
	$(CC) -o $@ python-tester.o $(CFLAGS) $(LIBS) $(INCLUDES)

%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $<

%.so: %.c
	$(CC) -shared -fPIC $< -o $@

clean:
	rm dgemm-splat dgemm-simple zgemm-splat python-tester *.o *.so *.optrpt

