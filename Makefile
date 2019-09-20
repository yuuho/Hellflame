.PHONY: clean uninstall install debug update

clean:
	find ./ -name '__pycache__' | xargs rm -rf
	find ./ -name '*.pyc' | xargs rm -f
	rm -rf hellfire.egg-info

uninstall:
	pip uninstall -y hellfire

install:
	pip install git+https://github.com/yuuho/Hellfire

debug:
	pip install -e .

update:
	pip install git+https://github.com/yuuho/Hellfire -U