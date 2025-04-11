CC = clang
CFLAGS = -Wall -Wextra -O2

TARGET = server
SRC = server.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)
	@./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean
