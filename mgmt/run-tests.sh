#!/usr/bin/env bash

echo "*** TEST RUN COMMENCING"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"
source "${PRAXONS}/gmalloc.sh"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

anybar yellow
pushd $PROJECT_PATH && \
    rm -rf ./build ./dist && \
    mkdir -p ./build ./dist && \
    pushd ./build && \
        anybar white && \
        cmake .. -Wno-dev -DCMAKE_INSTALL_PREFIX="${PROJECT_PATH}/dist" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
        anybar yellow && \
        make install && \
            anybar white && \
            ./imread_tests --success --durations yes --abortx 10 && anybar green || anybar red

# cd $PROJECT_PATH
popd
popd
echo "*** TEST RUN COMPLETE"