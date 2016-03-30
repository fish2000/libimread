#!/usr/bin/env bash
set -x

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}
: ${PROJECT:=$(dirname $THISDIR)}

source "${PRAXONS}/download.sh"
source "${PRAXONS}/urlcache.sh"

# N.B. do not use "master.zip" URLs as they all hash to the same fucking thing
deps="${PROJECT}/deps"
iod_dst="${deps}/iod"
iod_url="https://github.com/matt-42/iod/archive/f6c3f863a05e7938adf97d510886e1afe512facb.zip"

pushd $deps

    # what exists already?
    iod_tmp="${iod_dst}.old_temporary"
    [[ -d $iod_dst ]] && mv $iod_dst $iod_tmp
    
    set +x
    
    # fetch and expand anew
    fetch_and_expand $iod_url $iod_dst
    
    # cmakin out on the dancefloor
    CLANG_BIN="$(brew --prefix llvm)/bin"
    CC="${CLANG_BIN}/clang"
    CXX="${CLANG_BIN}/clang++"
    CFLAGS="-std=c99"
    CXXFLAGS="-std=c++14 -stdlib=libc++"
    
    pushd $iod_dst
        mkdir "build" && pushd "build"
            cmake .. -Wno-dev -DCMAKE_INSTALL_PREFIX=/tmp
            make -j4
        popd # build
        
    popd # iod_dst

popd # deps

# we're all the way here -- kill the old stuff
rm -rf $iod_tmp

echo "IM FINISHED"
