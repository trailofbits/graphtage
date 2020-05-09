# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile graphtage.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: graphtage.rst
graphtage.rst:
	# graphtage.py, edits.py, and tree.py are all merged into the main graphtage module by __init__.py,
	# so we should not generate separate submodules for them:
	sphinx-apidoc -f -e -M -T -o . ../graphtage ../graphtage/graphtage.py ../graphtage/edits.py ../graphtage/tree.py

.PHONY: clean
clean:
	rm -rf _build graphtage*.rst