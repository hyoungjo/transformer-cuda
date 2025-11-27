# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

SUBDIRS := cpp cuda

.PHONY: all clean $(SUBDIRS)

# ------------------------------------------------------------
# Build Rules
# ------------------------------------------------------------

all: $(SUBDIRS)

# -C: change directory
$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
