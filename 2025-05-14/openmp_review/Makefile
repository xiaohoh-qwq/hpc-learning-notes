CC = gcc
CFLAGS = -Wall -fopenmp -g

SRC = main.c square.c
OBJ = $(SRC:.c=.o)

TARGET = omp_test

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
