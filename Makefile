
all: clean multi

multi:
		mgmt/parallel-run-tests.sh

solo:
		mgmt/run-tests.sh

guard:
		mgmt/clean.sh
		GMALLOC=1 mgmt/parallel-run-tests.sh

re:
		@test -d build || echo "Can't resume: no build folder\n"
		@test -d build && pushd build && make -j4 && popd

test:
		@test -d build || echo "Can't run tests: no build folder\n"
		@test -d build && pushd build && ctest -j4 -D Experimental --output-on-failure && popd

clean:
		mgmt/clean.sh

.PHONY: all multi solo clean guard
