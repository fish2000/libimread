#!/usr/bin/env python
#
#       scan-test-data.py
#
#       Scan test data folder and write out a header file
#       For, like, 
#       (c) 2015 Alexander Bohn, All Rights Reserved
#

from __future__ import print_function
from os import chdir, listdir, getcwd
from os.path import dirname

datadir = "../data"

include_tpl = u"""
#ifndef IMREAD_TESTDATA_HPP_
#define IMREAD_TESTDATA_HPP_

#include <string>

namespace im {
    
    static const std::string basedir = "%(basedir)s";
    %(jpg)s
    %(jpeg)s
    %(png)s
    %(tif)s
    %(tiff)s
}

#endif /// IMREAD_TESTDATA_HPP_
"""

filetype_tpl = u"""
    static const int num_%(filetype)s = %(num)d;
    static const std::string %(filetype)s[] = {
    %(filestr)s
    };"""

def include(**kwargs):
    """ Required args:
            basedir:    string, path to base file directory (like duh)
            jpgs:       string, output from filetype() for jpg files
            jpegs:      string, output from filetype() for jpeg files
                                (N.B. jpg != jpeg)
            pngs:       string, output from filetype() for png files
            tifs:       string, output from filetype() for tif files
            tiffs:      string, output from filetype() for tiff files
                                (likely empty)
    """
    return include_tpl % kwargs

def filetype(suffix, files):
    """ Required Args:
            suffix:     string, file suffix (e.g. 'jpg', 'png' etc)
            files:      list, strings of actual file names per suffix
    """
    return filetype_tpl % dict(
        filestr=u",\n".join([u"    \"%s\"" % f for f in files]),
        filetype=suffix,
        num=len(files))

if __name__ == '__main__':
    chdir(dirname(__file__))
    chdir(datadir)
    basedir = getcwd()
    files = listdir(basedir)
    
    filetpls = dict()
    filemap = dict(
        jpg    =[f for f in files if f.endswith('jpg')],
        jpeg   =[f for f in files if f.endswith('jpeg')],
        png    =[f for f in files if f.endswith('png')],
        tif    =[f for f in files if f.endswith('tif')],
        tiff   =[f for f in files if f.endswith('tiff')])
    
    for suffix, filelist in filemap.iteritems():
        filetpls.update({ suffix: filetype(suffix, filelist) })
    filetpls.update({ 'basedir': basedir })
    print(include(**filetpls))
