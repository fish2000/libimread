
all: clean multi

multi:
		COVERAGE=OFF mgmt/parallel-run-tests.sh

coverage:
		COVERAGE=ON mgmt/parallel-run-tests.sh

solo:
		mgmt/run-tests.sh

guard:
		GMALLOC=1 mgmt/parallel-run-tests.sh

re:
		@test -d build || echo "Can't resume: no build folder\n"
		@test -d build && pushd build && make -j8 && popd

test:
		@test -d build || echo "Can't run tests: no build folder\n"
		@test -d build && pushd build && ctest -j8 -D Experimental --output-on-failure && popd

checkall:
		@test -d build || echo "Can't run tests: no build folder\n"
		@test -d build && pushd build && ./imread_tests --success --durations yes --abortx 10 && popd

check:
		@test -d build || echo "Can't run tests: no build folder\n"
		@test -d build && pushd build && ./imread_tests --durations yes --abortx 20 && popd

scantests:
		tests/scripts/generate-test-filemap.py > apps/TestDataViewer/TestDataViewer/TestData.plist

clean:
		mgmt/clean.sh

.PHONY: all multi solo clean guard
