#!/usr/bin/env bash

echo "*** CLEANUP IS NIGH"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

#rm -f ./tests/data/include/test_data.hpp
pushd $PROJECT_PATH && \
    rm -rf ./build ./dist
popd
