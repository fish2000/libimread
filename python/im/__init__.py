# -*- coding: utf-8 -*-
# Copyright (C) 2012-2016, Alexander Böhn <fish2000@gmail.com>
# License: MIT (see COPYING.MIT file)

from im import (
    _byteorder, _byteordermark,
    formats, format_info,
    
    detect,
    structcode_parse,
    
    Buffer,
    Image,
    Array,
    Batch)

from im import butteraugli as _small_butter_eye
from compat.show import show
from compat.to_PIL import to_PIL

def mimetype(suffix):
    if not suffix in format_info:
        return None
    return format_info[suffix].get('mimetype')

def butteraugli(lhs, rhs):
    lhs_capsule = lhs.encapsulate()
    rhs_capsule = rhs.encapsulate()
    return _small_butter_eye(lhs_capsule, rhs_capsule)

butteraugli.__doc__ = _small_butter_eye.__doc__