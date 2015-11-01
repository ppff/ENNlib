.PHONY: all reset examples $(SUBDIRS)

SUBDIRS := $(wildcard */.)

all: reset examples

reset:
	@reset
	@echo '***********************************'
	@echo '**** Compiling ENNlib examples ****'
	@echo '***********************************'

examples:
	@for dir in $(SUBDIRS); do \
		echo ' >>> Compiling example '$$dir; \
		$(MAKE) -C $$dir build clean; \
	done
