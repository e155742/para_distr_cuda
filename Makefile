CC = gcc
size = 1024
CFLAGS = -O3 -Wall -std=c11 -pedantic-errors -DSIZE=$(size)
print = false
ifeq ($(print), true)
CFLAGS += -DPRINT_VALUE -lm
endif
SRCCPU = matrix_cpu.c
OBJCPU = $(SRCCPU:.c=.o)
PROG = cpu

all: $(SRCCPU)
	$(CC) $(CFLAGS) -c -o $(OBJCPU) $(SRCCPU)
	$(CC) $(CFLAGS) $(OBJCPU) -o $(PROG)

clean:
	rm -f $(OBJCPU) $(PROG)

