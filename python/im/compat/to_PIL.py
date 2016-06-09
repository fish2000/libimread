
from __future__ import print_function

import numpy

def to_PIL(image, **options):
    from PIL import Image
    ia = numpy.array(image).transpose(1, 0, 2)
    return Image.fromarray(ia, **options)