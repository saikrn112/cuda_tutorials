main: kernels.o main.o
	nvcc kernels.o main_3.o -o ass3

kernels.o: kernels.cu kernels.h
	nvcc -c kernels.cu

main.o: main_3.cu
	nvcc -c main_3.cu

clean:
	rm -rf *.o ass3
