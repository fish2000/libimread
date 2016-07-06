/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <iod/json.hh>
#include <libimread/IO/tga.hh>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(TGAFormat);
    
    namespace detail { /// formerly 'TGA_pvt'
        
        enum tga_image_type {
            TYPE_NODATA         = 0,    ///< image with no data (why even spec it?)
            TYPE_PALETTED       = 1,    ///< paletted RGB
            TYPE_RGB            = 2,    ///< can include alpha
            TYPE_GRAY           = 3,    ///< can include alpha
            TYPE_PALETTED_RLE   = 9,    ///< same as PALETTED but run-length encoded
            TYPE_RGB_RLE        = 10,   ///< same as RGB but run-length encoded
            TYPE_GRAY_RLE       = 11    ///< same as GRAY but run-length encoded
        };
        
        enum tga_flags {
            FLAG_X_FLIP     = 0x10,   ///< right-left image
            FLAG_Y_FLIP     = 0x20    ///< top-down image
        };
        
        /// Targa file header:
        
        typedef struct {
            uint8_t idlen;              ///< image comment length
            uint8_t cmap_type;          ///< palette type
            uint8_t type;               ///< image type (see tga_image_type)
            uint16_t cmap_first;        ///< offset to first entry
            uint16_t cmap_length;       ///<
            uint8_t cmap_size;          ///< palette size
            uint16_t x_origin;          ///<
            uint16_t y_origin;          ///<
            uint16_t width;             ///< image width
            uint16_t height;            ///< image height
            uint8_t bpp;                ///< bits per pixel
            uint8_t attr;               ///< attribs (alpha bits and \ref tga_flags)
        } tga_header;
        
        /// TGA 2.0 file footer:
        
        typedef struct {
            uint32_t ofs_ext;           ///< offset to the extension area
            uint32_t ofs_dev;           ///< offset to the developer directory
            char signature[18];         ///< file signature string
        } tga_footer;
        
        /// TGA 2.0 developer directory entry:
        
        typedef struct {
            uint16_t tag;               ///< tag
            uint32_t ofs;               ///< byte offset to the tag data
            uint32_t size;              ///< tag data length
        } tga_devdir_tag;
        
        /// ... this is used in the extension area:
        
        enum tga_alpha_type {
            TGA_ALPHA_NONE              = 0,    ///< no alpha data included
            TGA_ALPHA_UNDEFINED_IGNORE  = 1,    ///< can ignore alpha
            TGA_ALPHA_UNDEFINED_RETAIN  = 2,    ///< undefined, but should be retained
            TGA_ALPHA_USEFUL            = 3,    ///< useful alpha data is present
            TGA_ALPHA_PREMULTIPLIED     = 4     ///< alpha is pre-multiplied (arrrgh!)
            // values 5-127 are reserved
            // values 128-255 are unassigned
        };
    
    }  /* namespace detail */
    
    
    std::unique_ptr<Image> TGAFormat::read(byte_source* src, ImageFactory* factory,
                                           options_map const& opts) {
        
        std::unique_ptr<Image> output = factory->create(bit_depth, height, width, channels);
        return output;
    }
    
    
    void PPMFormat::write(Image& input, byte_sink* output,
                          options_map const& opts) {
        /// DO IT DOUG
        bool dither = opts.cast<bool>("tga:dither", false);
        bool use_rle = opts.cast<bool>("tga:compress", false);
        float gamma_value = opts.cast<bool>("tga:gamma", 1.0);
        std::string targa_id = opts.cast<std::string>("tga:id", "");
        std::size_t targa_idlen = 0;
        
        /// prepare Targa header struct
        detail::tga_header tga = { 0 };
        tga.type = input.planes() <= 2 ? detail::tga_image_type::TYPE_GRAY :
                               use_rle ? detail::tga_image_type::TYPE_RGB_RLE :
                                         detail::tga_image_type::TYPE_RGB;
        tga.bpp = input.planes() * 8;
        tga.width = input.width();
        tga.height = input.height();
        tga.x_origin = 0; /// GENERALLY UNUSED
        tga.y_origin = 0; /// GENERALLY UNUSED
        
        tga.idlen = std::min(targa_id.size(), (std::size_t)255);
        targa_idlen = tga.idlen;
        
        /// This next line is a fucking sweet OIIO one-liner trick:
        if (input.planes() % 2 == 0) {
                                /// WE HAVE ALPHA:
            tga.attr = 8;       /// 8 bits of alpha
        }
        /// Add Y_FLIP flag if we're using RLE:
        if (use_rle) {
            tga.attr |= detail::tga_flags::FLAG_Y_FLIP;
        }
        
        /// TGAs are little-endian, soooo:
        if (detail::bigendian()) {
            detail::swap_endian(&tga.cmap_type);
            detail::swap_endian(&tga.type);
            detail::swap_endian(&tga.cmap_first);
            detail::swap_endian(&tga.cmap_length);
            detail::swap_endian(&tga.cmap_size);
            detail::swap_endian(&tga.x_origin);
            detail::swap_endian(&tga.y_origin);
            detail::swap_endian(&tga.width);
            detail::swap_endian(&tga.height);
            detail::swap_endian(&tga.bpp);
            detail::swap_endian(&tga.attr);
        }
        
        /// OIIO SEZ:
        /// "due to struct packing, we may get a corrupt header if we just dump the
        ///  struct to the file; to adress that, write every member individually"
        /// ... here we actually write out the header, member-by-member:
        if (!output->write(&tga.idlen) ||
            !output->write(&tga.cmap_type) ||
            !output->write(&tga.type) ||
            !output->write(&tga.cmap_first) ||
            !output->write(&tga.cmap_length) ||
            !output->write(&tga.cmap_size) ||
            !output->write(&tga.x_origin) ||
            !output->write(&tga.y_origin) ||
            !output->write(&tga.width) ||
            !output->write(&tga.height) ||
            !output->write(&tga.bpp) ||
            !output->write(&tga.attr))
        {
            imread_raise(CannotWriteError,
                "Failure writing TGA header");
        }
        
        if (targa_idlen) {
            if (!output->write(targa_id.c_str()))
        }
    }
    
    
}
