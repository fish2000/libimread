# Copyright (C) 2012-2016, Alexander Böhn <fish2000@gmail.com>
# License: MIT (see COPYING.MIT file)

from .imread import (
    imread, imload,
    imread_from_blob,
    imload_from_blob,
    imread_multi,
    imload_multi,
    
    imsave,
    imsave_multi,
    imwrite,
    imwrite_multi,
    
    detect_format,
    supports_format)