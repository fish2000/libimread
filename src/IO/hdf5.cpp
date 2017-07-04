/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>
#include <vector>
#include <hdf5.h>
#include <H5LTpublic.h>

#include <iod/json.hh>

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/file.hh>
#include <libimread/errors.hh>
#include <libimread/IO/hdf5.hh>

#if H5Dcreate_vers == 2
#define IM_H5D_CREATE(file_id, name, dtype, space_id)                   \
        H5Dcreate2(file_id, name, dtype, space_id,                      \
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
#else
#define IM_H5D_CREATE(file_id, name, dtype, space_id)                   \
        H5Dcreate(file_id, name, dtype, space_id,                       \
                  H5P_DEFAULT)
#endif

#if H5Dopen_vers == 2
#define IM_H5D_OPEN(file_id, name)                                      \
        H5Dopen2(file_id,                                               \
                 name, H5P_DEFAULT)
#else
#define IM_H5D_OPEN(file_id, name)                                      \
        H5Dopen(file_id,                                                \
                name)
#endif

#if H5Eset_auto_vers == 2
#define IM_H5E_SET_AUTO(flag) H5Eset_auto(flag, nullptr, nullptr)
#else
#define IM_H5E_SET_AUTO(flag) H5Eset_auto(nullptr, nullptr)
#endif

#define IM_H5F_CREATE(filepath) H5Fcreate(filepath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)


namespace im {
    
    DECLARE_FORMAT_OPTIONS(HDF5Format);
    
    using filesystem::path;
    using filesystem::NamedTemporaryFile;
    using namespace H5;
    
    class H5MemoryBuffer : public H5::H5File {
        public:
            static const unsigned OPEN_RW      =  H5LT_FILE_IMAGE_OPEN_RW;
            static const unsigned DONT_COPY    =  H5LT_FILE_IMAGE_DONT_COPY;
            static const unsigned DONT_RELEASE =  H5LT_FILE_IMAGE_DONT_RELEASE;
        
        public:
            H5MemoryBuffer(void* buffer, std::size_t size, unsigned flags)
                :H5File()
                {
                    hid_t idx = H5LTopen_file_image(buffer, size, flags);
                    if (idx < 0) {
                        imread_raise(HDF5IOError,
                            "H5MemoryBuffer constructor:",
                            "H5LTopen_file_image failed.");
                    }
                    p_setId(idx);
                }
    };
    
    const unsigned kDefaultFlags = H5MemoryBuffer::OPEN_RW &
                                   H5MemoryBuffer::DONT_COPY &
                                   H5MemoryBuffer::DONT_RELEASE;
    
    std::unique_ptr<Image> HDF5Format::read(byte_source* src,
                                            ImageFactory* factory,
                                            options_map const& opts) {
        
        /// load the raw sources' full data, and set some options:
        bytevec_t bytevec = src->full_data();
        path h5imagepath = opts.cast<path>("hdf5:path",
                                     path("/image/raster"));
        std::string name = opts.cast<std::string>("hdf5:name",
                                     std::string("imread-data"));
        
        /// HDF5 error status
        herr_t status;
        
        /// Open a data buffer as an HDF5 store (née "file image") for fast reading:
        hid_t file_id = H5LTopen_file_image(&bytevec[0], bytevec.size(), kDefaultFlags);
        if (file_id < 0) {
            imread_raise(CannotReadError,
                "Error opening HDF5 in-memory data as a file image for reading");
        }
        
        /// Open the image dataset within the HDF5 store:
        hid_t dataset_id = IM_H5D_OPEN(file_id, name.c_str());
        
        if (dataset_id < 0) {
            imread_raise(CannotReadError,
                "Error opening named HDF5 dataset for reading from in-memory HDF5 image");
        }
        
        /// set up an array to hold the dimensions of the image,
        /// which we read in from the in-memory HDF5 store:
        constexpr std::size_t NDIMS = 3;
        hid_t space_id = H5Dget_space(dataset_id);
        hsize_t dims[NDIMS];
        H5Sget_simple_extent_dims(space_id, dims, nullptr);
        
        /// Allocate a new unique-pointer-wrapped image instance:
        std::unique_ptr<Image> output = factory->create(8,  dims[0],
                                                            dims[1],
                                                            dims[2]);
        
        /// read typed data into the internal data buffer
        /// of the new image, from the HDF5 dataset in question:
        status = H5Dread(dataset_id, H5T_NATIVE_UCHAR,
                                     H5S_ALL, H5S_ALL,
                                     H5P_DEFAULT,
                                     output->rowp_as<uint8_t>(0));
        
        if (status < 0) {
            imread_raise(CannotReadError,
                "Error reading bytes into image from HDF5 dataset");
        }
        
        /// close HDF5 handles
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
        /// return the image pointer
        return output;
    }
    
    void HDF5Format::write(Image& input,
                           byte_sink* output,
                           options_map const& opts) {
        
        path h5imagepath = opts.cast<path>("hdf5:path",
                                     path("/image/raster"));
        std::string name = opts.cast<std::string>("hdf5:name",
                                     std::string("imread-data"));
        
        /// this wraps a call to H5Eset_auto(…), which I am not sure
        /// what that is all about, really; it’s just here:
        IM_H5E_SET_AUTO(H5E_DEFAULT);
        
        NamedTemporaryFile tf(".hdf5");
        std::string tpth = tf.str();
        herr_t status;
        
        hid_t file_id = IM_H5F_CREATE(tpth.c_str());
        
        if (file_id < 0) {
            imread_raise(CannotWriteError,
                "Error opening temporary HDF5 file for writing");
        }
        
        /// stow the image dimensions in an hsize_t array,
        /// and create two dataspaces based on that array:
        constexpr std::size_t NDIMS = 3;
        
        hsize_t dimensions[NDIMS] = { static_cast<hsize_t>(input.dim(0)),
                                      static_cast<hsize_t>(input.dim(1)),
                                      static_cast<hsize_t>(input.dim(2)) };
        
        hsize_t flattened[1] = { static_cast<hsize_t>(input.dim(0) *
                                                      input.dim(1) *
                                                      input.dim(2)) };
        
        /// space_id    --> “dataspace”: possibly strided dimensional data;
        /// memspace_id --> “memoryspace”: flattened 1-D view of dataspace;
        /// … the first arg to H5Screate_simple() is the rank (aka “ndims”)
        hid_t space_id      = H5Screate_simple(NDIMS, dimensions, nullptr);
        hid_t memspace_id   = H5Screate_simple(1,     flattened,  nullptr);
        
        /// try creating a new dataset --
        hid_t dataset_id = IM_H5D_CREATE(file_id, name.c_str(),
                                         H5T_NATIVE_UCHAR,
                                         space_id);
        
        /// -- and if that didn't work, try opening an existant one:
        if (dataset_id < 0) {
            dataset_id = IM_H5D_OPEN(file_id, name.c_str());
        }
        
        /// --- if we *still* lack a valid dataset_id, throw it up:
        if (dataset_id < 0) {
            imread_raise(CannotWriteError,
                "Could not create or open a file dataset while writing HDF5 data");
        }
        
        /// create attributes to save metadata: nbits, ndims, dim(x), stride(x)
        hid_t attspace_id = H5Screate(H5S_SCALAR);
        
        if (attspace_id < 0) {
            imread_raise(CannotWriteError,
                "Error creating an attribute dataspace while writing HDF5 data");
        }
        
        #define CreateAttribute(name, type)                                                     \
            H5Acreate(dataset_id, name, type, attspace_id, H5P_DEFAULT, H5P_DEFAULT)
        
        #define CreateIntAttribute(name)                                                        \
            CreateAttribute(name, H5T_NATIVE_INT)
        
        hid_t attr_nbits   = CreateIntAttribute("nbits");
        hid_t attr_ndims   = CreateIntAttribute("ndims");
        hid_t attr_dim0    = CreateIntAttribute("dim0");
        hid_t attr_dim1    = CreateIntAttribute("dim1");
        hid_t attr_dim2    = CreateIntAttribute("dim2");
        hid_t attr_stride0 = CreateIntAttribute("stride0");
        hid_t attr_stride1 = CreateIntAttribute("stride1");
        hid_t attr_stride2 = CreateIntAttribute("stride2");
        
        #undef CreateAttribute
        #undef CreateIntAttribute
        
        const int val_nbits   = input.nbits();
        const int val_ndims   = input.ndims();
        const int val_dim0    = input.dim(0);
        const int val_dim1    = input.dim(1);
        const int val_dim2    = input.dim(2);
        const int val_stride0 = input.stride(0);
        const int val_stride1 = input.stride(1);
        const int val_stride2 = input.stride(2);
        
        /// actually write the image data
        status = H5Dwrite(dataset_id,   H5T_NATIVE_UCHAR,
                          memspace_id,
                          space_id,     H5P_DEFAULT,
                          (const void *)input.rowp(0));
        
        if (status < 0) {
            imread_raise(CannotWriteError,
                "Error writing HDF5 dataset bytes (via memory dataspaces) to disk");
        }
        
        /// write the attributes
        
        #define WriteIntAttribute(name)                                                         \
            H5Awrite(attr_##name, H5T_NATIVE_INT, &val_##name)
        
        WriteIntAttribute(nbits);
        WriteIntAttribute(ndims);
        WriteIntAttribute(dim0);
        WriteIntAttribute(dim1);
        WriteIntAttribute(dim2);
        WriteIntAttribute(stride0);
        WriteIntAttribute(stride1);
        WriteIntAttribute(stride2);
        
        #undef WriteIntAttribute
        
        /// close attribute dataspace and attribute handles
        H5Sclose(attspace_id);
        
        #define CloseAttribute(name)                                                            \
            H5Aclose(attr_##name)
        
        CloseAttribute(nbits);
        CloseAttribute(ndims);
        CloseAttribute(dim0);
        CloseAttribute(dim1);
        CloseAttribute(dim2);
        CloseAttribute(stride0);
        CloseAttribute(stride1);
        CloseAttribute(stride2);
        
        #undef CloseAttribute
        
        /// close all HDF5 handles
        H5Sclose(memspace_id);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
        /// read all binary data back from the temporary file
        std::unique_ptr<FileSource> readback(new FileSource(tf.filepath));
        bytevec_t reread_bytevec = readback->full_data();
        
        /// rewrite the binary data using the target output byte sink
        output->write((const void*)&reread_bytevec[0],
                                    reread_bytevec.size());
        output->flush();
    }
    
}
