#!/usr/bin/env bash

echo "*** SYNCING TEST PICTURE DATA TO ~/Pictures/libimread-test-data"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"

PROJECT_PATH="/Users/fish/Dropbox/libimread"
EXTERNAL_PATH="/Users/fish/Pictures/libimread-test-data"

: ${COLOR_TRACE:="ON"}
: ${VERBOSE:="ON"}

pushd ${EXTERNAL_PATH}/..
    rm -rf libimread-test-data && \
    mkdir libimread-test-data
popd

pushd $PROJECT_PATH
    mcp "tests/data/*.*" "~/Pictures/libimread-test-data/#1.#2"
popd

echo "*** EXTERNAL SYNC COMPLETE"