/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <iod/json.hh>
#include <libimread/IO/bmp.hh>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(BMPFormat);
    
    namespace {
        
        void flippixels(byte* row, const int n) {
            for (int i = 0; i != n; row += 3, ++i) {
                byte b = row[0];
                byte g = row[1];
                byte r = row[2];
                row[0] = r;
                row[1] = g;
                row[2] = b;
            }
        }
        
        void color_expand(const std::vector<byte>& color_table, byte* row,
                          const int w) {
            // We are expanding in-place --
            // This means that we must process the row backwards,
            // which is slightly awkward, but works correctly
            std::vector<byte>::const_iterator pbegin = color_table.begin();
            for (int i = w - 1; i >= 0; --i) {
                std::copy(
                    pbegin + 4 * row[i],
                    pbegin + 4 * row[i] + 3,
                    row + 3 * i);
            }
        }
        
        uint32_t pow2(uint32_t n) {
            if (n <= 0) { return 1; }
            return 2 * pow2(n - 1);
        }
        
        /// XXX: While these read***() funcs look generic,
        /// they are primarily only of use in exactly one codec
        /// (which I think is bmp.cpp but I could be wrong,
        /// fuck opening another buffer to check dogg I am lazy)
        uint8_t read8(byte_source& s) {
            byte out;
            if (s.read(&out, 1) != 1) {
                imread_raise(CannotReadError,
                    "im::read8(): File ended prematurely");
            }
            return out;
        }
        
        inline uint16_t read16_le(byte_source& s) {
            uint8_t b0 = read8(s);
            uint8_t b1 = read8(s);
            return (uint16_t(b1) << 8) | uint16_t(b0);
        }
        
        inline uint32_t read32_le(byte_source& s) {
            uint16_t s0 = read16_le(s);
            uint16_t s1 = read16_le(s);
            return (uint32_t(s1) << 16) | uint32_t(s0);
        }
        
    }
    
    std::unique_ptr<Image> BMPFormat::read(byte_source* src,
                                           ImageFactory* factory,
                                           const options_map& opts)  {
        char magic[2];
        
        if (src->read(reinterpret_cast<byte*>(magic), 2) != 2) {
            imread_raise(CannotReadError, "File is empty");
        }
        
        if (magic[0] != 'B' || magic[1] != 'M') {
            imread_raise(CannotReadError,
                "Magic number not matched", "(this might not be a BMP file)");
        }
        
        const uint32_t size = read32_le(*src);
        (void)size;
        (void)read16_le(*src);
        (void)read16_le(*src);
        
        const uint32_t offset = read32_le(*src);
        const uint32_t header_size = read32_le(*src);
        (void)header_size;
        
        const uint32_t width = read32_le(*src);
        const uint32_t height = read32_le(*src);
        const uint16_t planes = read16_le(*src);
        
        if (planes != 1) {
            imread_raise(NotImplementedError, "planes should be 1");
        }
        
        const uint16_t bitsppixel = read16_le(*src);
        const uint32_t compression = read32_le(*src);
        
        if (compression != 0) {
            imread_raise(NotImplementedError, "Only uncompressed bitmaps are supported");
        }
        
        const uint32_t imsize = read32_le(*src);
        (void)imsize;
        const uint32_t hres = read32_le(*src);
        (void)hres;
        const uint32_t vres = read32_le(*src);
        (void)vres;
        const uint32_t n_colours = read32_le(*src);
        const uint32_t importantcolours = read32_le(*src);
        (void)importantcolours;
        
        if (bitsppixel != 8 && bitsppixel != 16 && bitsppixel != 24) {
            imread_raise(CannotReadError,
                FF("Bits per pixel is %i", bitsppixel),
                   " -- only bpp values of 8, 16, or 24 are supported");
        }
        
        const int depth = (bitsppixel == 16 ? -1 : 3);
        const int nbits = (bitsppixel == 16 ? 16 : 8);
        std::unique_ptr<Image> output(
            factory->create(nbits, height, width, depth));
        
        std::vector<byte> color_table;
        if (bitsppixel <= 8) {
            const uint32_t table_size = (n_colours == 0 ? pow2(bitsppixel) : n_colours);
            color_table.resize(table_size * 4);
            src->read_check(&color_table[0], table_size * 4);
        }
        
        src->seek_absolute(offset);
        
        const int bytes_per_row = width * (bitsppixel / 8);
        const int padding = (4 - (bytes_per_row % 4)) % 4;
        byte buf[4];
        
        for (unsigned int r = 0; r != height; ++r) {
            byte* rowp = output->rowp_as<byte>(height - r - 1);
            src->read_check(rowp, bytes_per_row);
            
            if (bitsppixel == 24) {
                flippixels(rowp, width);
            } else if (!color_table.empty()) {
                color_expand(color_table, rowp, width);
            }
            
            if (src->read(buf, padding) != unsigned(padding) && r != (height - 1)) {
                imread_raise(CannotReadError, "File ended prematurely while reading");
            }
        }
        
        return output;
    }
}
