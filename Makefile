CXX     := g++
CFLAGS  := -g -std=c++11 -Wall -O3

SRCDIR      := src
INCDIR      := include/galois
BUILDDIR    := build
BINDIR      := bin
EXAMPLEDIR  := example
TESTDIR     := test

SRC         := $(wildcard $(SRCDIR)/*.cc        $(SRCDIR)/*/*.cc)
EXAMPLESRC  := $(wildcard $(EXAMPLEDIR)/*.cc    $(EXAMPLEDIR)/*/*.cc)
TESTSRC     := $(wildcard $(TESTDIR)/*.cc       $(TESTDIR)/*/*.cc)

OBJ         := $(patsubst $(SRCDIR)/%.cc,       $(BUILDDIR)/%.o,        $(SRC))
EXAMPLEOBJ  := $(patsubst $(EXAMPLEDIR)/%.cc,   $(BUILDDIR)/example/%.o, $(EXAMPLESRC))
TESTOBJ     := $(patsubst $(TESTDIR)/%.cc,      $(BUILDDIR)/test/%.o,   $(TESTSRC))

EXAMPLE     := $(patsubst $(EXAMPLEDIR)/%.cc,   $(BINDIR)/%, $(EXAMPLESRC))
TEST        := $(patsubst $(TESTDIR)/%.cc,      $(BINDIR)/%, $(TESTSRC))

# platform detection
OS := $(shell uname -s)

ifeq ($(OS), Darwin)
	INC         := -I include
	LIB         := -lz -framework accelerate
else
	OPENBLASDIR := /opt/OpenBLAS # set /your_path/OpenBLAS
	INC         := -I include -I $(OPENBLASDIR)/include
	LIB         := -lz -L $(OPENBLASDIR)/lib
endif

.PHONY: all clean

all: $(EXAMPLE) $(TEST)

$(EXAMPLE): $(BINDIR)/%: $(BUILDDIR)/example/%.o $(OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(LIB) $< $(OBJ) -o $@

$(EXAMPLEOBJ): $(BUILDDIR)/example/%.o: $(EXAMPLEDIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

$(TEST): $(BINDIR)/%: $(BUILDDIR)/test/%.o $(OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(LIB) $< $(OBJ) -o $@

$(TESTOBJ): $(BUILDDIR)/test/%.o: $(TESTDIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

$(OBJ): $(BUILDDIR)/%.o: $(SRCDIR)/%.cc $(INCDIR)/%.h
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

$(BUILDDIR)/narray.o: $(INCDIR)/narray_functors.cc

clean:
	rm -r $(BUILDDIR)/*
	rm -r $(BINDIR)/*
