/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <errno.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/IO/hdf5.hh>
#include <H5Cpp.h>

namespace im {
    
    using filesystem::path;
    using namespace H5;
    
    class H5MemoryBuffer : public H5::H5File {
        
        public:
            H5MemoryBuffer(void *buffer, std::size_t size, unsigned flags)
                :H5File()
                {
                    hid_t idx = H5LTopen_file_image(buffer, size, flags);
                    if (idx < 0) {
                        throw FileIException("H5MemoryBuffer constructor:",
                                                 "H5LTopen_file_image failed");
                    }
                    p_setId(idx);
                }
    };
    
    const unsigned H5MemoryBuffer::OPEN_RW      =  H5LT_FILE_IMAGE_OPEN_RW;
    const unsigned H5MemoryBuffer::DONT_COPY    =  H5LT_FILE_IMAGE_DONT_COPY;
    const unsigned H5MemoryBuffer::DONT_RELEASE =  H5LT_FILE_IMAGE_DONT_RELEASE;
    
    const unsigned kDefaultFlags = H5MemoryBuffer::OPEN_RW &
                                   H5MemoryBuffer::DONT_COPY &
                                   H5MemoryBuffer::DONT_RELEASE;
    
    std::unique_ptr<Image> HDF5Format::read(byte_source *src,
                                            ImageFactory *factory,
                                            const options_map &opts) {
        
        std::vector<byte> data = src->full_data();
        path imagepath = opts.cast<path>("hdf5:path",
                                   path("/image/raster"));
        std::string nm = opts.cast<std::string>("hdf5:name",
                                   std::string("imread-data"));
        
        H5MemoryBuffer store(&data[0], data.size(), kDefaultFlags);
        Group group(store.openGroup(imagepath));
        std::unique_ptr<DataSet> dataset(new DataSet(
                                 group.openDataSet(nm)));
        
        DSetCreatPropList plist = dataset->getCreatePlist();
        
        H5T_class_t typeclass = dataset->getTypeClass();
        IntType inttype = dataset->getIntType();
        std::string byteorder_string;
        H5T_order_t byteorder = inttype.getOrder(byteorder_string);
        std::size_t size = inttype.getSize();
        
        std::unique_ptr<DataSpace> dataspace(new DataSpace(
                                   dataset->getSpace()));
        
        int rank = dataspace->getSimpleExtentNdims(); /// should be 3
        std::unique_ptr<hsize_t[]> DIMS(new hsize_t[rank]);
        int ndims = dataspace->getSimpleExtentDims(DIMS.get(), NULL);
        
        
        
    }

}
