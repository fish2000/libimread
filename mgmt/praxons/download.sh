#!/usr/bin/env bash
# download_to, expand_tarball_to, expand_zipwad_to, download_and_expand

download_to () {
    in_url="${1:?URL expected}"
    out_file="${2:?pathname expected}"
    success=0
    [[ -r $out_file ]] && echo "- Already exists: ${out_file}" && return 1
    echo "+ Retrieving remote file: ${in_url}"
    test ! -r $out_file && test -x `which wget` && wget --tries=40 --retry-connrefused $in_url -O $out_file && success=1
    test ! -r $out_file && test -x `which curl` && curl -L $in_url -o $out_file && success=1
    test ! -r $out_file && test -x `which http` && http -d $in_url -o $out_file && success=1
    [ $success == 0 ] && test -r $out_file && rm $out_file
    [ $success == 0 ] && echo "- Couldn't download. Tried: wget, curl, httpie" && return 1
    echo "+ Downloaded to: ${out_file}"
}

expand_tarball_to () {
    in_tarball="${1:?tarball expected}"
    out_directory="${2:?pathname expected}"
    [[ ! -r $in_tarball ]] && echo "- Can't read tarball: ${in_tarball}" && return 1
    [[ -d $out_directory ]] && rm -rf $out_directory
    mkdir -p $out_directory
    echo "+ Expanding tarball: $(basename ${in_tarball})"
    echo "+ Expansion destination: ${out_directory}"
    tar xzf $in_tarball --strip-components=1 --directory=$out_directory
}

expand_zipwad_to () {
    in_zipwad="${1:?zipwad expected}"
    out_directory="${2:?pathname expected}"
    tmp_directory="$(mktemp -d -t XXXXX)"
    [[ ! -r $in_zipwad ]] && echo "- Can't read zipwad: ${in_zipwad}" && return 1
    [[ -d $out_directory ]] && rm -rf $out_directory
    echo "+ Unzipping zipwad: $(basename ${in_zipwad})"
    echo "+ Unzip destination: ${out_directory}"
    unzip -q -d $tmp_directory $in_zipwad
    expanded_directory=("$tmp_directory"/*)
    if (( ${#tmp_directory[@]} == 1 )) && [[ -d $tmp_directory ]]; then
        mv "${tmp_directory}"/* $out_directory && rm -rf $tmp_directory
    else
        echo "- Unzip failed, deleting temporary files: ${tmp_directory}"
        [[ -d $tmp_directory ]] && rm -rf $tmp_directory
        return 1
    fi
}

download_and_expand () {
    url="${1:?URL expected}"
    url_basename="$(basename ${url})"
    url_suffix="${url_basename##*.}"
    destination_directory="${2:-${url_basename%%.*}}"
    tmp_directory="$(mktemp -d -t XXXXX)"
    tmp_archive="${tmp_directory}/${url_basename}"
    download_to $url $tmp_archive || return 1
    [[ ${url_suffix,,} == *zip ]] \
        && expand_zipwad_to $tmp_archive $destination_directory
    [[ ${url_suffix,,} != *zip ]] \
        && expand_tarball_to $tmp_archive $destination_directory
    rm $tmp_archive
    [[ -d $tmp_directory ]] && rm -rf $tmp_directory
}
