/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>
#include <vector>
#include <array>

#include <iod/json.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/file.hh>
#include <libimread/errors.hh>
#include <libimread/metadata.hh>
#include <libimread/IO/hdf5.hh>

#include <H5Cpp.h>
#include <hdf5.h>
#include <H5LTpublic.h>
#include <libimread/ext/h5deets.hh>

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

#define IM_H5F_CREATE(filepath)     H5Fcreate(filepath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
#define IM_H5S_CREATE(rank, dims)   H5Screate_simple(rank, dims, nullptr)

namespace im {
    
    DECLARE_FORMAT_OPTIONS(HDF5Format);
    
    namespace detail {
        constexpr std::size_t kDefaultFlags = H5LT_FILE_IMAGE_OPEN_RW &
                                              H5LT_FILE_IMAGE_DONT_COPY &
                                              H5LT_FILE_IMAGE_DONT_RELEASE;
    }
    
    using filesystem::path;
    using filesystem::NamedTemporaryFile;
    
    std::unique_ptr<Image> HDF5Format::read(byte_source* src,
                                            ImageFactory* factory,
                                            Options const& opts) {
        
        /// Internal options for our HDF5 data storage:
        path h5imagepath = opts.cast<path>("hdf5:path", path(options.datapath));
        std::string name = opts.cast<std::string>("hdf5:name", options.dataname);
        
        /// Open data buffer as an HDF5 memory store (née "file image"):
        hid_t file_id = H5LTopen_file_image(src->data(),
                                            src->size(),
                                            detail::kDefaultFlags);
        if (file_id < 0) {
            imread_raise(CannotReadError,
                "Error opening HDF5 in-memory data as a file image for reading");
        }
        
        /// Open the image dataset within the HDF5 memory store:
        hid_t dataset_id = IM_H5D_OPEN(file_id, name.c_str());
        
        if (dataset_id < 0) {
            imread_raise(CannotReadError,
                "Error opening named HDF5 dataset for reading from in-memory HDF5 image");
        }
        
        /// Set up a std::array to hold the dimensions of the image,
        /// which we read in from the HDF5 memory store:
        constexpr std::size_t NDIMS = 3;
        hid_t dataspace_id = H5Dget_space(dataset_id);
        
        // std::array<hsize_t, NDIMS> dims, maxdims;
        // H5Sget_simple_extent_dims(dataspace_id, dims.data(),
        //                                      maxdims.data());
        
        std::array<hsize_t, NDIMS> dims;
        H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);
        
        /// Allocate a new unique-pointer-wrapped im::Image instance:
        std::unique_ptr<Image> output = factory->create(8,  dims[1],
                                                            dims[0],
                                                            dims[2]);
        
        /// Read H5T_NATIVE_UCHAR data into the internal data buffer
        /// of the new image, from the HDF5 dataset in question:
        herr_t status = H5Dread(dataset_id, detail::typecode<byte>(),
                                            H5S_ALL, H5S_ALL,
                                            H5P_DEFAULT,
                                            output->rowp_as<byte>(0));
        
        if (status < 0) {
            imread_raise(CannotReadError,
                "Error reading bytes into image from HDF5 dataset");
        }
        
        /// If our newly-created im::Image supports metadata,
        /// read and store HDF5 attributes as such:
        if (Metadata* meta = dynamic_cast<Metadata*>(output.get())) {
            /// ATTRIBUTES!
            using detail::h5a_t;
            
            h5a_t nbits(dataset_id,   "nbits");
            h5a_t ndims(dataset_id,   "ndims");
            h5a_t dim0(dataset_id,    "dim0");
            h5a_t dim1(dataset_id,    "dim1");
            h5a_t dim2(dataset_id,    "dim2");
            h5a_t stride0(dataset_id, "stride0");
            h5a_t stride1(dataset_id, "stride1");
            h5a_t stride2(dataset_id, "stride2");
            
            int val_nbits   = nbits.typed_read<int>();
            int val_ndims   = ndims.typed_read<int>();
            int val_dim0    = dim0.typed_read<int>();
            int val_dim1    = dim1.typed_read<int>();
            int val_dim2    = dim2.typed_read<int>();
            int val_stride0 = stride0.typed_read<int>();
            int val_stride1 = stride1.typed_read<int>();
            int val_stride2 = stride2.typed_read<int>();
            
            meta->set("nbits",   std::to_string(val_nbits));
            meta->set("ndims",   std::to_string(val_ndims));
            meta->set("dim0",    std::to_string(val_dim0));
            meta->set("dim1",    std::to_string(val_dim1));
            meta->set("dim2",    std::to_string(val_dim2));
            meta->set("stride0", std::to_string(val_stride0));
            meta->set("stride1", std::to_string(val_stride1));
            meta->set("stride2", std::to_string(val_stride2));
        }
        
        /// Close HDF5 hid_t handle types:
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
        /// Return the im::Image
        return output;
    }
    
    void HDF5Format::write(Image& input,
                           byte_sink* output,
                           Options const& opts) {
        
        /// Internal options for our HDF5 data storage:
        path h5imagepath = opts.cast<path>("hdf5:path", path(options.datapath));
        std::string name = opts.cast<std::string>("hdf5:name", options.dataname);
        
        /// This wraps a call to H5Eset_auto(…), which I am not sure
        /// what that is all about, really; it’s just here:
        IM_H5E_SET_AUTO(H5E_DEFAULT);
        
        /// create a temporary HDF5 file for all writes to target:
        NamedTemporaryFile tf(suffix(true)); /// period = true
        
        std::string tpth = tf.str();
        hid_t file_id = IM_H5F_CREATE(tpth.c_str());
        
        if (file_id < 0) {
            imread_raise(CannotWriteError,
                "Could not open a temporary file for writing with HDF5 file I/O");
        }
        
        /// Stow the input image dimensions in an hsize_t std::array,
        /// and create two HDF5 dataspaces based on that array:
        constexpr std::size_t NDIMS = 3;
        
        std::array<hsize_t, NDIMS> dimensions{{
            static_cast<hsize_t>(input.dim(0)),
            static_cast<hsize_t>(input.dim(1)),
            static_cast<hsize_t>(input.dim(2))
        }};
        
        std::array<hsize_t, 1> flattened{{
            static_cast<hsize_t>(input.dim(0) *
                                 input.dim(1) *
                                 input.dim(2))
        }};
        
        /// space_id    --> “dataspace”: possibly strided dimensional data;
        /// memspace_id --> “memoryspace”: flattened 1-D view of dataspace;
        /// … the first arg to H5Screate_simple() is the rank (aka “ndims”)
        hid_t dataspace_id  = IM_H5S_CREATE(NDIMS, dimensions.data());
        hid_t memspace_id   = IM_H5S_CREATE(1,     flattened.data());
        
        /// Try creating a new dataset --
        hid_t dataset_id = IM_H5D_CREATE(file_id, name.c_str(),
                                         detail::typecode<byte>(),
                                         dataspace_id);
        
        /// -- If that didn't work, try opening an existant one:
        if (dataset_id < 0) {
            dataset_id = IM_H5D_OPEN(file_id, name.c_str());
        }
        
        /// --- If we *still* lack a valid dataset_id, throw it up:
        if (dataset_id < 0) {
            imread_raise(CannotWriteError,
                "Could not create or open an HDF5 dataset for writing with hyperslab I/O");
        }
        
        /// Store some basic im::Image properties of our image
        /// as HDF5 scalar integer-attributes, attached to the
        /// image dataset handle:
        {
            /// ATTRIBUTES!
            using detail::attspace_t;
            using detail::h5t_t;
            using detail::h5a_t;
            
            attspace_t attspace = attspace_t::scalar();
            h5t_t tc = detail::typecode<int>();
            
            h5a_t nbits(dataset_id,   "nbits",   attspace, tc);
            h5a_t ndims(dataset_id,   "ndims",   attspace, tc);
            h5a_t dim0(dataset_id,    "dim0",    attspace, tc);
            h5a_t dim1(dataset_id,    "dim1",    attspace, tc);
            h5a_t dim2(dataset_id,    "dim2",    attspace, tc);
            h5a_t stride0(dataset_id, "stride0", attspace, tc);
            h5a_t stride1(dataset_id, "stride1", attspace, tc);
            h5a_t stride2(dataset_id, "stride2", attspace, tc);
            
            /// actually write the image data
            herr_t status = H5Dwrite(dataset_id,   detail::typecode<byte>(),
                                     memspace_id,
                                     dataspace_id, H5P_DEFAULT,
                                    (const void*)input.rowp(0));
            
            if (status < 0) {
                imread_raise(CannotWriteError,
                    "Error writing HDF5 dataset bytes (via memory dataspaces) to disk");
            }
            
            nbits.typed_write(input.nbits());
            ndims.typed_write(input.ndims());
            dim0.typed_write(input.dim(0));
            dim1.typed_write(input.dim(1));
            dim2.typed_write(input.dim(2));
            stride0.typed_write(input.stride(0));
            stride1.typed_write(input.stride(1));
            stride2.typed_write(input.stride(2));
            
        } /// end of ATTRIBUTES! scope exit closes all the things!
        
        /// Close all HDF5 hid_t handle types:
        H5Sclose(memspace_id);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        
        /// Read binary data back from the temporary file:
        std::unique_ptr<FileSource> readback(new FileSource(tf.filepath));
        
        /// Rewrite all binary data using the target output byte sink:
        output->write(readback->data(), readback->size());
        output->flush();
    }
    
}
