#!/usr/bin/env bash

echo "*** TEST RUN COMMENCING"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"
source "${PRAXONS}/gmalloc.sh"

: ${PROJECT_PATH:="/Users/fish/Dropbox/libimread"}

: ${APPS:="ON"}
: ${COLOR_TRACE:="ON"}
: ${COVERAGE:="ON"}
: ${VERBOSE:="ON"}
: ${TERMINATOR:="ON"}
: ${TESTS:="ON"}
: ${PROCESSORS:=$(py 'multiprocessing.cpu_count()')}

pushd $PROJECT_PATH
    rm -rf ${PROJECT_PATH}/build ${PROJECT_PATH}/dist
    rm -f ${PROJECT_PATH}/default.profraw
    mkdir -p ${PROJECT_PATH}/build
    mkdir -p ${PROJECT_PATH}/dist
    
    pushd ./build
        cmake .. \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_INSTALL_PREFIX="${PROJECT_PATH}/dist" \
            -DIM_APPS=$APPS \
            -DIM_COLOR_TRACE=$COLOR_TRACE \
            -DIM_COVERAGE=$COVERAGE \
            -DIM_VERBOSE=$VERBOSE \
            -DIM_TERMINATOR=$TERMINATOR \
            -DIM_TESTS=$TESTS \
            -Wno-dev && \
        make -j$PROCESSORS install
        [[ $COVERAGE == ON ]] && make -j$PROCESSORS imread_tests_coverage
        [[ $COVERAGE == OFF ]] && ctest -j$PROCESSORS -D Experimental --output-on-failure
    
    popd
    rm -rf ./Testing ./default.profraw

popd

echo "*** TEST RUN COMPLETE"