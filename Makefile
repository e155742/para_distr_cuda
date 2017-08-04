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

all: clean cpu cuda

cpu: clean $(SRC)
	$(CC) $(CFLAGS) -x c -c -o $(OBJ) $(SRC)
	$(CC) $(CFLAGS) -o $(PROGCPU) $(OBJ)

cuda gpu: clean $(SRC)
	$(CUC) $(CUFLAGS) -c -o $(OBJ) $(SRC)
	$(CUC) $(CUFLAGS) -o $(PROGCUDA) $(OBJ)

clean:
	rm -f $(OBJ) $(PROGCPU) $(PROGCUDA)

