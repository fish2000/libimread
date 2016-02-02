
def filemapper(files):
    return dict(
        jpg    = [f for f in files if f.endswith('jpg')],
        jpeg   = [f for f in files if f.endswith('jpeg')],
        png    = [f for f in files if f.endswith('png')],
        tif    = [f for f in files if f.endswith('tif')],
        tiff   = [f for f in files if f.endswith('tiff')],
        hdf5   = [f for f in files if f.endswith('hdf5')],
        pvr    = [f for f in files if f.endswith('pvr')])
