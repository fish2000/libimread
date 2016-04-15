/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>
#include <vector>
#include <memory>

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/file.hh>
#include <libimread/errors.hh>
#include <libimread/IO/hdf5.hh>

#include <H5LTpublic.h>

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
                                            const options_map& opts) {
        
        /// load the raw sources' full data, and set some options:
        std::vector<byte> data = src->full_data();
        path h5imagepath = opts.cast<path>("hdf5:path",
                                     path("/image/raster"));
        std::string name = opts.cast<std::string>("hdf5:name",
                                     std::string("imread-data"));
        
        /// HDF5 error status
        herr_t status;
        
        /// Open a data buffer as an HDF5 store (née "file image") for fast reading:
        hid_t file_id = H5LTopen_file_image(&data[0], data.size(), kDefaultFlags);
        if (file_id < 0) {
            imread_raise(CannotReadError,
                "Error opening HDF5 binary data");
        }
        
        /// Open the image dataset within the HDF5 store:
        hid_t dataset_id;
        #if H5Dopen_vers == 2
            dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
        #else
            dataset_id = H5Dopen(file_id, name.c_str());
        #endif
        if (dataset_id < 0) {
            imread_raise(CannotReadError,
                "Error opening HDF5 dataset within data buffer");
        }
        
        /// set up an array to hold the dimensions of the image,
        /// which we read in from the in-memory HDF5 store:
        constexpr std::size_t NDIMS = 3;
        hid_t space_id = H5Dget_space(dataset_id);
        hsize_t dims[NDIMS];
        H5Sget_simple_extent_dims(space_id, dims, NULL);
        
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
                "Error reading HDF5 dataset");
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
                           const options_map& opts) {
        
        path h5imagepath = opts.cast<path>("hdf5:path",
                                        path("/image/raster"));
        std::string name = opts.cast<std::string>("hdf5:name",
                                        std::string("imread-data"));
        
        #if H5Eset_auto_vers == 2
            H5Eset_auto( H5E_DEFAULT, NULL, NULL );
        #else
            H5Eset_auto( NULL, NULL );
        #endif
        
        NamedTemporaryFile tf(".hdf5");
        herr_t status;
        // hid_t file_id = H5Fopen(tf.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT);
        // if (file_id < 0) {
        //     WTF("H5Fopen() failed, trying H5Fcreate() with:", tf.str());
        //     file_id = H5Fcreate(tf.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        // }
        hid_t file_id = H5Fcreate(tf.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            imread_raise(CannotWriteError,
                "Error opening temporary HDF5 file for writing");
        }
        
        /// stow the image dimensions in an hsize_t array,
        /// and create two dataspaces based on that array:
        constexpr std::size_t NDIMS = 3;
        hsize_t dims[NDIMS] = { static_cast<hsize_t>(input.dim(0)),
                                static_cast<hsize_t>(input.dim(1)),
                                static_cast<hsize_t>(input.dim(2)) };
        hsize_t mdims[1] = { static_cast<hsize_t>(input.dim(0) *
                                                  input.dim(1) *
                                                  input.dim(2)) };
        hid_t space_id      = H5Screate_simple(NDIMS, dims, NULL); /// first arg here is rank (aka NDIMS)
        hid_t memspace_id   = H5Screate_simple(1, mdims, NULL);
        
        /// try creating a new dataset --
        hid_t dataset_id;
        #if H5Dcreate_vers == 2
            dataset_id = H5Dcreate2(file_id, name.c_str(),
                                    H5T_NATIVE_UCHAR,
                                    space_id,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        #else
            dataset_id = H5Dcreate(file_id, name.c_str(),
                                   H5T_NATIVE_UCHAR,
                                   space_id,
                                   H5P_DEFAULT);
        #endif
        
        /// -- and if that didn't work, try opening an existant one:
        if (dataset_id < 0) {
            #if H5Dopen_vers == 2
                dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
            #else
                dataset_id = H5Dopen(file_id, name.c_str());
            #endif
        }
        if (dataset_id < 0) {
            imread_raise(CannotWriteError,
                "Error creating or opening dataset in temporary HDF5 file");
        }
        
        /// create attributes to save metadata: nbits, ndims, dim(x), stride(x)
        hid_t attspace_id = H5Screate(H5S_SCALAR);
        
        #define Attribute(name, type)                                                   \
            H5Acreate(dataset_id, name, type, attspace_id, H5P_DEFAULT, H5P_DEFAULT)
        
        hid_t attr_nbits   = Attribute("nbits",   H5T_NATIVE_INT);
        hid_t attr_ndims   = Attribute("ndims",   H5T_NATIVE_INT);
        hid_t attr_dim0    = Attribute("dim0",    H5T_NATIVE_INT);
        hid_t attr_dim1    = Attribute("dim1",    H5T_NATIVE_INT);
        hid_t attr_dim2    = Attribute("dim2",    H5T_NATIVE_INT);
        hid_t attr_stride0 = Attribute("stride0", H5T_NATIVE_INT);
        hid_t attr_stride1 = Attribute("stride1", H5T_NATIVE_INT);
        hid_t attr_stride2 = Attribute("stride2", H5T_NATIVE_INT);
        
        #undef Attribute
        
        const int val_nbits   = input.nbits();
        const int val_ndims   = input.ndims();
        const int val_dim0    = input.dim(0);
        const int val_dim1    = input.dim(1);
        const int val_dim2    = input.dim(2);
        const int val_stride0 = input.stride(0);
        const int val_stride1 = input.stride(1);
        const int val_stride2 = input.stride(2);
        
        /// actually write the image data
        status = H5Dwrite(dataset_id, H5T_NATIVE_UCHAR,
                          memspace_id, space_id,
                          H5P_DEFAULT,
                          (const void *)input.rowp(0));
        
        if (status < 0) {
            imread_raise(CannotWriteError,
                "Error writing to temporary HDF5 dataset");
        }
        
        /// write the attributes
        H5Awrite(attr_nbits,   H5T_NATIVE_INT, &val_nbits);
        H5Awrite(attr_ndims,   H5T_NATIVE_INT, &val_ndims);
        H5Awrite(attr_dim0,    H5T_NATIVE_INT, &val_dim0);
        H5Awrite(attr_dim1,    H5T_NATIVE_INT, &val_dim1);
        H5Awrite(attr_dim2,    H5T_NATIVE_INT, &val_dim2);
        H5Awrite(attr_stride0, H5T_NATIVE_INT, &val_stride0);
        H5Awrite(attr_stride1, H5T_NATIVE_INT, &val_stride1);
        H5Awrite(attr_stride2, H5T_NATIVE_INT, &val_stride2);
        
        /// close attribute dataspace and attribute handles
        H5Sclose(attspace_id);
        H5Aclose(attr_nbits);
        H5Aclose(attr_ndims);
        H5Aclose(attr_dim0);
        H5Aclose(attr_dim1);
        H5Aclose(attr_dim2);
        H5Aclose(attr_stride0);
        H5Aclose(attr_stride1);
        H5Aclose(attr_stride2);
        
        /// close all HDF5 handles
        H5Sclose(memspace_id);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
        /// read the binary data back from the temporary file,
        /// and write it out to the output byte sink
        std::unique_ptr<FileSource> readback(new FileSource(tf.filepath));
        std::vector<byte> data = readback->full_data();
        output->write((const void*)&data[0], data.size());
        output->flush();
        
        /// clean up file resources
        readback->close();
        readback.reset(nullptr);
        bool removed = tf.remove();
        if (!removed) {
            removed = tf.filepath.remove();
            if (!removed) {
                WTF("Couldn't remove NamedTemporaryFile after HDF5 output:",
                    tf.filepath.str());
            }
        }
    }
    
}
