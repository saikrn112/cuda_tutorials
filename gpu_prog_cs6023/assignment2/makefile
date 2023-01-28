main: kernels.o main.o
	nvcc kernel.o main.o -o msort

kernels.o: kernel.cu kernel.h
	nvcc -c kernel.cu

main.o: main.cu
	nvcc -c main.cu

clean:
	rm -rf *.o msort
