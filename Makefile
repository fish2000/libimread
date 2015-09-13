
all: clean multi

multi:
		mgmt/parallel-run-tests.sh

solo:
		mgmt/run-tests.sh

re:
		@test -d build || echo "Can't resume: no build folder\n"
		@test -d build && pushd build && make && popd
#		@test -d build && pushd build && anybar white && make && anybar blue || anybar yellow && popd

clean:
		mgmt/clean.sh

.PHONY: all multi solo clean
