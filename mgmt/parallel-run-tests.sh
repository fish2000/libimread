#!/usr/bin/env bash

echo "*** TEST RUN COMMENCING"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"
source "${PRAXONS}/gmalloc.sh"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

: ${APPS:="OFF"}
: ${COLOR_TRACE:="ON"}
: ${VERBOSE:="ON"}
: ${TERMINATOR:="ON"}
: ${PROCESSORS:=$(py 'multiprocessing.cpu_count()')}

pushd $PROJECT_PATH
    rm -rf ./build ./dist
    mkdir -p ./build ./dist
    pushd ./build
        cmake .. \
            -DCMAKE_INSTALL_PREFIX="${PROJECT_PATH}/dist" \
            -DIM_APPS=$APPS \
            -DIM_COLOR_TRACE=$COLOR_TRACE \
            -DIM_VERBOSE=$VERBOSE \
            -DIM_TERMINATOR=$TERMINATOR \
            -Wno-dev && \
        make -j$PROCESSORS install && \
        ctest -j$PROCESSORS -D Experimental --output-on-failure

popd
rm -rf ./Testing

popd

echo "*** TEST RUN COMPLETE"