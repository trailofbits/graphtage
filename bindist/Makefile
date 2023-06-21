GRAPHTAGE_VERSION=$(shell graphtage --version 2>&1 | sed "s/Graphtage version //")
DIST_VERSION=$(shell uname | tr '[:upper:]' '[:lower:]')-$(shell uname -m | tr '[:upper:]' '[:lower:]')
DIST_NAME=graphtage-$(GRAPHTAGE_VERSION)-$(DIST_VERSION)
DIST_FILE=$(DIST_NAME).zip

.PHONY: $(DIST_FILE)
$(DIST_FILE):
	pyinstaller -F -y --name graphtage graphtage_bin.py
	@rm -rf $(DIST_NAME)
	mkdir $(DIST_NAME)
	cp dist/graphtage $(DIST_NAME)/
	cp -p ../README.md $(DIST_NAME)/
	cp -p ../LICENSE $(DIST_NAME)/
	zip -r $(DIST_FILE) $(DIST_NAME)
	rm -rf $(DIST_NAME)

.PHONY: dist-name
dist-name:
	@echo $(DIST_FILE)

.PHONY: clean
clean:
	rm -rf graphtage.spec dist build $(DIST_FILE) $(DIST_NAME)
