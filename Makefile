CC = gcc
CUC = nvcc
size = 1024
thread = 16
COMMOMFLAGS = -O3 -lm -DSIZE=$(size)
CFLAGS = -std=c11 -Wall -pedantic-errors $(COMMOMFLAGS)
CUFLAGS = -arch=sm_30 -DTHREAD=$(thread) $(COMMOMFLAGS)
SRC = matrix.cu
OBJ = $(SRC:.cu=.o)
PROGCPU = cpu
PROGCUDA = cuda

all: cleanobj cpu cuda

cpu: cleanobj $(SRC)
	$(CC) $(CFLAGS) -x c -c -o $(OBJ) $(SRC)
	$(CC) $(CFLAGS) -o $(PROGCPU) $(OBJ)

cuda gpu: cleanobj $(SRC)
	$(CUC) $(CUFLAGS) -c -o $(OBJ) $(SRC)
	$(CUC) $(CUFLAGS) -o $(PROGCUDA) $(OBJ)

clean: cleanobj
	rm -f $(PROGCPU) $(PROGCUDA)

cleanobj:
	rm -f $(OBJ)
