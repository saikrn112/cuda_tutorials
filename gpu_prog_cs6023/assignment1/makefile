main: kernels.o main.o
	nvcc kernels.o main.o -o transpose

kernels.o: kernels.cu kernels.h
	nvcc -c kernels.cu

main.o: main.cu
	nvcc -c main.cu

clean: 
	rm -rf *.o transpose
