
SOURCES := $(wildcard tip*/tip-*.md)
KERNELS := $(SOURCES:%.md=%)

pdfs: ${KERNELS}
	@echo ""

%:
	pandoc -o $@.pdf $@.md


