#!/usr/bin/env bash

echo "*** TEST RUN COMMENCING"

PROJECT_PATH="/Users/fish/Dropbox/libimread"

pushd $PROJECT_PATH && \
    rm -rf ./build ./dist && \
    mkdir -p ./build ./dist && \
    pushd ./build && \
    cmake .. -Wno-dev -DCMAKE_INSTALL_PREFIX=./dist && \
    make install && \
        ./imread_tests --success --durations yes --abortx 10

# cd $PROJECT_PATH
popd
popd
echo "*** TEST RUN COMPLETE"