.PHONY: all

all:
	pyinstaller -F -y --name graphtage graphtage_bin.py
	cp dist/graphtage .

.PHONY: clean
clean:
	rm -rf graphtage graphtage.spec dist build
