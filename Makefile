# Define the compiler
NVCC=nvcc

# Define the flags to pass to the compiler
# NVCCFLAGS=-arch=sm_35

# Define the directories for source and object files
SRCDIR=src
OBJDIR=obj
MKDIR_P = mkdir -p $(OBJDIR)

# Define the source files to compile
SOURCES=$(wildcard $(SRCDIR)/*.cu)

# Define the object files to create
OBJECTS=$(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Define the rule for compiling source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(MKDIR_P)
	$(NVCC) -dc -o $@ $<

# Define the linking rule
main.out: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@

# Define the "clean" rule
clean:
	rm -rf $(OBJDIR)