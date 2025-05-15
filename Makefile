CC = gcc
MPICC = mpicc

CFLAGS = -I.
MPIFLAGS = -I.

EXEC_SERIAL = serial
EXEC_MPI = parallel parallel_1
ALL_EXEC = $(EXEC_MPI) $(EXEC_SERIAL)

ifdef debug
    CFLAGS += -DDEBUG
    MPIFLAGS += -DDEBUG
endif

all: $(ALL_EXEC)


serial: serial.c config.h
	$(CC) $(CFLAGS) -o $@ $^

parallel: conjugate-solver.c config.h
	$(MPICC) $(MPIFLAGS) -o $@ $^ -lm

parallel_1: parallel.c config.h
	$(MPICC) $(MPIFLAGS) -o $@ $^


# Clean rule
clean:
	rm -f $(ALL_EXEC) *.txt