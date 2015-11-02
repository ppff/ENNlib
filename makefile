.PHONY : clean doc build examples

SRCDIR   = src
INCDIR   = includes
BUILDDIR = build

SOURCES = $(shell echo $(SRCDIR)/*.cpp)
HEADERS = $(shell echo $(INCDIR)/*.h $(INCDIR)/*.hpp)
OBJECTS = $(SOURCES:.cpp=.o)

LIBNAME = libENN
TARGET = $(BUILDDIR)/$(LIBNAME).so

COMPILER= g++
CPPFLAGS= -I$(INCDIR) -std=c++11 -fPIC -g -Wall -O3 -fopenmp
LDFLAGS = -shared

all: reset build clean

reset : 
	@reset
	@echo '**************************'
	@echo '**** Compiling ENNlib ****'
	@echo '**************************'

build: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "[Info] Compiling library"
	@$(COMPILER) $(CPPFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "[Info] Library compiled"

%.o : %.cpp
	@echo '[Info] (Compiling '$<')'
	@$(COMPILER) $< $(CPPFLAGS) -c -o $(basename $<).o

clean:
	@echo "[Info] Cleaning object files"
	@rm -f $(OBJECTS)

doc:
	@echo "[Info] Generating documentation"
	@doxygen Doxyfile
	@echo "[Info] Documentation generated"

examples:
	@echo '****************************'
	@echo '**** Compiling examples ****'
	@echo '****************************'
	@$(MAKE) -C ./examples examples
