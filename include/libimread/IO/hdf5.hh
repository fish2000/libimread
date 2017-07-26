/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_HDF5_HH_
#define LIBIMREAD_IO_HDF5_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class HDF5Format : public ImageFormatBase<HDF5Format> {
        
        public:
            static constexpr std::size_t kDimensions = 3;
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            /// \x0d\x0a\x1a\x0a
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x89\x48\x44\x46", 4)
                },
                _suffixes = { "hdf5", "h5", "hdf" },
                _mimetype = "image/hdf5",
                _dataname = "imread-data",
                _datapath = "/image/raster",
                _dimensions = kDimensions
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts) override;
    };
    
    namespace format {
        using H5 = HDF5Format;
        using HDF5 = HDF5Format;
    }
    
}

/*
 * Via: http://www.digitalpreservation.gov/formats/fdd/fdd000229.shtml
 *  ... which that looks like a useful resource in general
 */

#endif /// LIBIMREAD_IO_HDF5_HH_
