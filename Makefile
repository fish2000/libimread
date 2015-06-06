
multi:
		mgmt/parallel-run-tests.sh

solo:
		mgmt/run-tests.sh

re:
		@test -d build || echo "Can't resume: no build folder\n"
		@test -d build && pushd build && make && popd

clean:
		mgmt/clean.sh

all: multi

.PHONY: all multi solo clean