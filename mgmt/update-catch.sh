#!/usr/bin/env bash

: ${THISDIR:=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
: ${PRAXONS:="${THISDIR}/praxons"}

source "${PRAXONS}/download.sh"
source "${PRAXONS}/urlcache.sh"

url="https://raw.githubusercontent.com/philsquared/Catch/master/single_include/catch.hpp"

