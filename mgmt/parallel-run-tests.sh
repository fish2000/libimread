#!/usr/bin/env bash

echo "*** TEST RUN COMMENCING"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"
source "${PRAXONS}/gmalloc.sh"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

anybar yellow
pushd $PROJECT_PATH
    rm -rf ./build ./dist
    mkdir -p ./build ./dist
    pushd ./build
        anybar white
        cmake .. \
            -Wno-dev \
            -DCMAKE_INSTALL_PREFIX=./dist
        anybar yellow
        make -j4 install && \
            anybar white && \
        ctest -j4 -D Experimental --output-on-failure && \
            anybar green || anybar red

popd
rm -rf ./Testing

popd

echo "*** TEST RUN COMPLETE"