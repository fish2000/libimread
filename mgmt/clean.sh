#!/usr/bin/env bash

echo "*** CLEANUP IS NIGH"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

pushd $PROJECT_PATH &> /dev/null
    #rm -f ./tests/data/include/test_data.hpp
    rm -f ./include/libimread/symbols.hpp
    rm -rf ./build ./dist

popd &> /dev/null

pushd $TMPDIR
    rm -rf ./test-* ./write-* *.imdata
popd
