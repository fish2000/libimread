
clean: clean-pyc clean-so

distclean: clean-all-pyc clean-pycache clean-so clean-build-artifacts

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

clean-build-artifacts:
	rm -rf build dist imread.egg-info

buildext: distclean
	python setup.py build_ext --inplace

build: distclean
	python setup.py build_ext --inplace

nosetests:
	NOSE_REDNOSE=1 python runtests.py

alltests:
	py.test --color=yes --verbose -r a --full-trace

test:
	py.test -k 'not hybridimage' --color=yes --verbose -r a --full-trace

profile:
	py.test --color=yes --verbose -r a --profile-svg

upload:
	python setup.py sdist upload

.PHONY: clean distclean dist buildext upload
