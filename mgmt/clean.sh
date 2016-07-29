#!/usr/bin/env bash

echo "*** CLEANUP IS NIGH"

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
source "${PRAXONS}/anybar.sh"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

pushd $PROJECT_PATH &> /dev/null
    #rm -f ./tests/data/include/test_data.hpp
    #rm -f ./include/libimread/symbols.hpp
    rm -rf ./build ./dist ./check

popd &> /dev/null

pushd $TMPDIR
    rm -rf ./test-* ./write-* *.imdata *.imr
popd

anybar blue