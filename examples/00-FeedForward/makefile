.PHONY : clean build

EXAMPLENAME = example00

LIB_BIN_DIR = ../../build
LIB_INC_DIR = ../../includes

SOURCES = $(shell echo *.cpp)
HEADERS = $(shell echo *.h *.hpp)
OBJECTS = $(SOURCES:.cpp=.o)

COMPILER= g++
CPPFLAGS= -I$(LIB_INC_DIR) -std=c++11 -g -Wall
LDFLAGS = -L$(LIB_BIN_DIR) -lENN -Wl,-rpath=$(LIB_BIN_DIR)

all: reset build clean start

reset : 
	@reset
	@echo '*********'
	@echo 'Compiling '$(EXAMPLENAME)
	@echo '*********'

build: $(EXAMPLENAME)

$(EXAMPLENAME): $(OBJECTS)
	@$(COMPILER) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Example compiled"

%.o : %.cpp
	@$(COMPILER) $< $(CPPFLAGS) -c -o $(basename $<).o

clean:
	@echo "Cleaning object files"
	@rm -f $(OBJECTS)

start:
	@echo
	@echo 'Launching example...'
	@echo '-----------------'
	@./$(EXAMPLENAME)
	@echo '-----------------'
	@echo 'Example over.'
	@echo
