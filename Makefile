CXX 	:= g++
CFLAGS 	:= -g -std=c++11 -Wall -O3

SRCDIR 		:= src
BUILDDIR 	:= build
BINDIR 		:= bin
EXAMPLEDIR 	:= example
TESTDIR 	:= test

SRC 		:= $(wildcard $(SRCDIR)/*.cc 		$(SRCDIR)/*/*.cc)
EXAMPLESRC 	:= $(wildcard $(EXAMPLEDIR)/*.cc 	$(EXAMPLEDIR)/*/*.cc)
TESTSRC		:= $(wildcard $(TESTDIR)/*.cc 		$(TESTDIR)/*/*.cc)

OBJ 		:= $(patsubst $(SRCDIR)/%.cc, 		$(BUILDDIR)/%.o, 		$(SRC))
EXAMPLEOBJ 	:= $(patsubst $(EXAMPLEDIR)/%.cc, 	$(BUILDDIR)/example/%.o, $(EXAMPLESRC))
TESTOBJ 	:= $(patsubst $(TESTDIR)/%.cc, 		$(BUILDDIR)/test/%.o, 	$(TESTSRC))

EXAMPLE 	:= $(patsubst $(EXAMPLEDIR)/%.cc, 	$(BINDIR)/%, $(EXAMPLESRC))
TEST		:= $(patsubst $(TESTDIR)/%.cc, 		$(BINDIR)/%, $(TESTSRC))

INC 		:= -I include
LIB 		:= -framework accelerate -lz

.PHONY: all clean

all: $(EXAMPLE) $(TEST)

$(EXAMPLE): $(BINDIR)/%: $(BUILDDIR)/example/%.o $(OBJ)
	$(CXX) $(CFLAGS) $(LIB) $< $(OBJ) -o $@

$(EXAMPLEOBJ): $(EXAMPLESRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

$(TEST): $(BINDIR)/%: $(BUILDDIR)/test/%.o $(OBJ)
	$(CXX) $(CFLAGS) $(LIB) $< $(OBJ) -o $@

$(TESTOBJ): $(TESTSRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(INC) -c $< -o $@

clean:
	rm -r $(BUILDDIR)/*
	rm -r $(BINDIR)/*

