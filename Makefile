CXX := g++
CFLAGS := -g -std=c++11 -Wall

SRCDIR := src
BUILDDIR := build

SRC := $(wildcard $(SRCDIR)/*.cc $(SRCDIR)/*/*.cc)
OBJ := $(patsubst $(SRCDIR)/%.cc, $(BUILDDIR)/%.o, $(SRC))
INC := -I include
LIB := -framework accelerate

TARGET := bin/main

.PHONY: clean

$(TARGET): $(OBJ)
	$(CXX) $(CFLAGS) $(LIB) $(OBJ) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

clean: $(OBJ)
	rm $(OBJ)
