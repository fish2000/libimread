
clean: clean-pyc clean-so

distclean: clean-all-pyc clean-pycache clean-so clean-build-artifacts clean-generators

ext: buildext test

prepare: build test

dist: ext upload

clean-pyc:
	find . -name \*.pyc -print -delete

clean-all-pyc:
	find . -name \*.pyc -print -delete

clean-pycache:
	find . -path \*__pycache__\* -print -delete

clean-so:
	rm -f im/*.so

clean-generators:
	rm -f im/resources/generators/*.a im/resources/generators/*.h

clean-build-artifacts:
	rm -rf build dist imread.egg-info

buildext: distclean
	python setup.py build_ext --inplace

build: distclean
	python setup.py build_ext --inplace

nosetests:
	NOSE_REDNOSE=1 python runtests.py

alltests:
	py.test --color=yes --verbose -r a --durations=20 --tb=long

test:
	py.test -k 'not hybridimage and not singular' --color=yes --verbose -r a --showlocals --durations=20 --tb=long

profile:
	py.test --color=yes --verbose -r a --profile-svg

upload:
	python setup.py sdist upload

.PHONY: clean distclean dist buildext upload
