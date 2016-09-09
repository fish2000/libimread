
# TESTS = batch compat jupyter_repr planar_ops read readwrite structcode
# TESTS = batch compat planar_ops read readwrite structcode
TESTS = structcode readwrite read planar_ops compat batch

TESTDIR = im/tests
PYTESTARGS = --color=yes --verbose -r a --showlocals --durations=10 --tb=long
PYTESTS = $(TESTS:%=$(TESTDIR)/test_%.py)

.PHONY: clean distclean dist buildext upload test testall alltests im/tests/test_%.py

clean: clean-pyc clean-pycache clean-so

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
	rm -f im/resources/generators/*.a im/resources/generators/*.h im/resources/generators/*.dylib

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

testall:
	py.test -k 'not hybridimage and not singular and not jupyter' $(PYTESTARGS)

test:
	@$(foreach pytest, $(PYTESTS), echo "py.test $(pytest) $(PYTESTARGS)" && py.test $(pytest) $(PYTESTARGS);)

upload:
	python setup.py sdist upload

