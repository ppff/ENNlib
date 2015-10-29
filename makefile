.PHONY : clean doc

COMPILER= g++
CPPFLAGS= -fPIC -g -Wall
LDFLAGS = -shared

SRCDIR   = src
INCDIR   = includes
BUILDDIR = build

SOURCES = $(shell echo $(SRCDIR)/*.cpp)
HEADERS = $(shell echo $(INCDIR)/*.h $(INCDIR)/*.hpp)
OBJECTS = $(SOURCES:.cpp=.o)

LIBNAME = libENN
TARGET = $(BUILDDIR)/$(LIBNAME).so

all: build clean

build: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Compiling library"
	$(COMPILER) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Library compiled"

clean:
	@echo "Cleaning object files and target"
	@rm -f $(OBJECTS) $(TARGET)

doc:
	@echo "Generating documentation"
	doxygen Doxyfile
	@echo "Documentation generated"
