#!/usr/bin/env bash

: ${GMALLOC:=0}

if (( ${GMALLOC} == 1 )); then
    echo ">>>>>> RUNNING WITH GUARD MALLOC LOADED (libgmalloc.dylib)"
    export DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib"
else
    echo "*** Running without guard malloc (libgmalloc.dylib)"
fi