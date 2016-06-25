
import im
import numpy

def imread(filename, formatstr, flags):
    return (numpy.array(im.Array(filename)).transpose(1, 0, 2), '')

def imread_from_blob(blob, formatstr, flags):
    return numpy.array(im.Array(blob, as_blob=True)).transpose(1, 0, 2)

def imread_multi(filename, formatstr, flags):
    raise NotImplementedError("_imread.imread_multi is as-of-yet unsupported")

def imsave(filename, formatstr, array, opts):
    imbuffer = im.Buffer.frompybuffer(array.transpose(1, 0, 2))
    image = im.Array.frombuffer(imbuffer)
    image.write(filename, as_blob=False, options=opts)

def imsave_multi(filename, formatstr, arrays, opts):
    raise NotImplementedError("_imread.imsave_multi is as-of-yet unsupported")

def detect_format(filename_or_blob, is_blob):
    return im.detect(filename_or_blob, bool(is_blob))

def supports_format(formatstr):
    return formatstr.lower() in im.formats