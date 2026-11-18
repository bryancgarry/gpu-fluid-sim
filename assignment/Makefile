all: assignment.cu
	nvcc -O2 assignment.cu -o assignment.exe -lcufft -lcurand

clean:
	rm -f assignment.exe *.pgm