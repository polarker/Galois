CXX := g++
CFLAGS := -g -std=c++11 -Wall

SRCDIR := src
BUILDDIR := build
BINDIR := bin
EXAMPLEDIR := example

SRC := $(wildcard $(SRCDIR)/*.cc $(SRCDIR)/*/*.cc)
OBJ := $(patsubst $(SRCDIR)/%.cc, $(BUILDDIR)/%.o, $(SRC))
EXAMPLESRC := $(wildcard $(EXAMPLEDIR)/*.cc $(EXAMPLEDIR)/*/*.cc)
EXAMPLEOBJ := $(patsubst $(EXAMPLEDIR)/%.cc, $(BUILDDIR)/example/%.o, $(EXAMPLESRC))
EXAMPLE := $(patsubst $(EXAMPLEDIR)/%.cc, $(BINDIR)/%, $(EXAMPLESRC))
INC := -I include
LIB := -framework accelerate -lz

.PHONY: all clean

all: $(EXAMPLE)

$(EXAMPLE): $(EXAMPLEOBJ) $(OBJ)
	$(CXX) $(CFLAGS) $(LIB) $(OBJ) -o $@

$(EXAMPLEOBJ): $(EXAMPLESRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

clean:
	rm $(OBJ) $(EXAMPLEOBJ) $(EXAMPLE)

