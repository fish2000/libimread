
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

checkall:
		@test -d build || echo "Can't run tests: no build folder\n"
		@test -d build && pushd build && imread_tests --success --durations yes --abortx 10 && popd

check:
		@test -d build || echo "Can't run tests: no build folder\n"
		@test -d build && pushd build && imread_tests --durations yes --abortx 20 && popd

scantests:
		tests/scripts/generate-test-filemap.py > apps/TestDataViewer/TestDataViewer/TestData.plist

clean:
		mgmt/clean.sh

.PHONY: all multi solo clean guard
