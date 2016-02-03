#!/usr/bin/env bash

echo "*** TEST RUN COMMENCING"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"
source "${PRAXONS}/gmalloc.sh"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

pushd $PROJECT_PATH
    rm -rf ./build ./dist
    mkdir -p ./build ./dist
    pushd ./build
        cmake .. \
            -Wno-dev \
            -DCMAKE_INSTALL_PREFIX="${PROJECT_PATH}/dist"
        make -j4 install && \
        ctest -j4 -D Experimental --output-on-failure && \
            anybar green || anybar red

popd
rm -rf ./Testing

popd

echo "*** TEST RUN COMPLETE"