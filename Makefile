clean:
	find ./ -name '__pycache__' | xargs rm -rf
	find ./ -name '*.pyc' | xargs rm -f

uninstall:
	pip uninstall -y hellfire

install:
	pip install git+https://github.com/yuuho/Hellfire