/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <csetjmp>
#include <algorithm>
#include <vector>
#include <array>

#include <iod/json.hh>

#include <libimread/errors.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/pixels.hh>
#include <libimread/seekable.hh>
#include <libimread/ext/exif.hh>

extern "C" {
#include <jpeglib.h>
}

namespace im {
    
    DECLARE_FORMAT_OPTIONS(JPEGFormat);
    
    namespace {
        
        const std::size_t buffer_size = JPEGFormat::options.buffer_size;
        
        struct jpeg_source_adaptor {
            jpeg_source_mgr mgr;
            byte_source* s;
            byte* __restrict__ buf;
            
            jpeg_source_adaptor(byte_source* s);
            ~jpeg_source_adaptor() { delete[] buf; }
        };
        
        struct jpeg_dst_adaptor {
            jpeg_destination_mgr mgr;
            byte_sink* s;
            byte* __restrict__ buf;
            
            jpeg_dst_adaptor(byte_sink* s);
            ~jpeg_dst_adaptor() { delete[] buf; }
        };
        
        void nop(j_decompress_ptr cinfo) {}
        void nop_dst(j_compress_ptr cinfo) {}
        
        boolean fill_input_buffer(j_decompress_ptr cinfo) {
            jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
            adaptor->mgr.next_input_byte = adaptor->buf;
            adaptor->mgr.bytes_in_buffer = adaptor->s->read(adaptor->buf, buffer_size);
            return (boolean)true;
        }
        
        void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
            if (num_bytes <= 0) { return; }
            jpeg_source_adaptor* adaptor = reinterpret_cast<jpeg_source_adaptor*>(cinfo->src);
            while (num_bytes > long(adaptor->mgr.bytes_in_buffer)) {
                num_bytes -= adaptor->mgr.bytes_in_buffer;
                fill_input_buffer(cinfo);
            }
            adaptor->mgr.next_input_byte += num_bytes;
            adaptor->mgr.bytes_in_buffer -= num_bytes;
        }
        
        boolean empty_output_buffer(j_compress_ptr cinfo) {
            jpeg_dst_adaptor* adaptor = reinterpret_cast<jpeg_dst_adaptor*>(cinfo->dest);
            adaptor->s->write(
                adaptor->buf,
                buffer_size);
            adaptor->mgr.next_output_byte = adaptor->buf;
            adaptor->mgr.free_in_buffer = buffer_size;
            return (boolean)true;
        }
        
        void flush_output_buffer(j_compress_ptr cinfo) {
            jpeg_dst_adaptor* adaptor = reinterpret_cast<jpeg_dst_adaptor*>(cinfo->dest);
            adaptor->s->write(
                adaptor->buf,
                adaptor->mgr.next_output_byte - adaptor->buf);
            adaptor->s->flush();
        }
        
        jpeg_source_adaptor::jpeg_source_adaptor(byte_source* s)
            :s(s) 
            {
                buf = new byte[buffer_size];
                mgr.next_input_byte = buf;
                mgr.bytes_in_buffer = 0;
                mgr.init_source = nop;
                mgr.fill_input_buffer = fill_input_buffer;
                mgr.skip_input_data = skip_input_data;
                mgr.resync_to_restart = jpeg_resync_to_restart;
                mgr.term_source = nop;
            }
        
        jpeg_dst_adaptor::jpeg_dst_adaptor(byte_sink* s)
            :s(s)
            {
                buf = new byte[buffer_size];
                mgr.next_output_byte = buf;
                mgr.free_in_buffer = buffer_size;
                mgr.init_destination = nop_dst;
                mgr.empty_output_buffer = empty_output_buffer;
                mgr.term_destination = flush_output_buffer;
            }
        
        inline J_COLOR_SPACE color_space(int components) {
            if (components == 1) { return JCS_GRAYSCALE; }
            if (components == 3) { return JCS_RGB; }
            if (components == 4) { return JCS_CMYK; }
            imread_raise(CannotReadError,
                "\tim::(anon)::color_space() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
             FF("\tim::(anon)::color_space() got:    `components` = (int){ %i }", components),
                "\tim::(anon)::color_space() needs:  `components` = (int){ 1, 3, 4 }");
        }
        
        struct jpeg_decompress_holder {
            
            jpeg_decompress_holder() {
                jpeg_create_decompress(&info);
            }
            
            ~jpeg_decompress_holder() {
                jpeg_finish_decompress(&info);
                jpeg_destroy_decompress(&info);
            }
            
            void start() {
                jpeg_read_header(&info, (boolean)true);
                jpeg_start_decompress(&info);
            }
            
            int height() const      { return info.output_height; }
            int width() const       { return info.output_width; }
            int components() const  { return info.output_components; }
            int scanline() const    { return info.output_scanline; }
            
            JSAMPARRAY allocate_samples(int width, int depth, int height = 1) {
                return (*info.mem->alloc_sarray)((j_common_ptr)&info, JPOOL_IMAGE,
                                                                      width * depth,
                                                                      height);
            }
            
            void read_samples(JSAMPARRAY samples, int idx = 1) {
                jpeg_read_scanlines(&info, samples, idx);
            }
            
            public:
                jpeg_decompress_struct info;
        };
        
        struct jpeg_compress_holder {
            
            jpeg_compress_holder() {
                jpeg_create_compress(&info);
            }
            
            ~jpeg_compress_holder() {
                jpeg_finish_compress(&info);
                jpeg_destroy_compress(&info);
            }
            
            void start() {
                jpeg_start_compress(&info, (boolean)true);
            }
            
            void set_defaults() {
                jpeg_set_defaults(&info);
            }
            
            void set_quality(std::size_t quality) {
                if (quality > 100) { quality = 100; }
                jpeg_set_quality(&info, quality, (boolean)false);
            }
            
            int height() const          { return info.image_height; }
            int width() const           { return info.image_width; }
            int next_scanline() const   { return info.next_scanline; }
            
            void set_height(int height)         { info.image_height = height; }
            void set_width(int width)           { info.image_width = width; }
            void set_components(int components) { info.input_components = components;
                                                  info.in_color_space = color_space(components); }
            
            void write_scanlines(JSAMPLE** rows, int idx = 1) {
                jpeg_write_scanlines(&info, rows, idx);
            }
            
            public:
                jpeg_compress_struct info;
        };
        
        struct error_mgr {
            error_mgr();
            mutable struct jpeg_error_mgr pub;
            mutable jmp_buf setjmp_buffer;
            mutable char error_message[JMSG_LENGTH_MAX];
            
            bool has_error() const {
                return setjmp(this->setjmp_buffer);
            }
        };
        
        void err_long_jump(j_common_ptr cinfo) {
            error_mgr* err = reinterpret_cast<error_mgr*>(cinfo->err);
            (*cinfo->err->format_message)(cinfo, err->error_message);
            longjmp(err->setjmp_buffer, 1);
        }
        
        error_mgr::error_mgr() {
            jpeg_std_error(&pub);
            pub.error_exit = err_long_jump;
            error_message[0] = 0;
        }
        
        template <typename Iterator>
        uint16_t parse_size(Iterator it) {
            /// extract size of an EXIF byte region:
            Iterator siz0 = std::next(it, 2);
            Iterator siz1 = std::next(it, 3);
            return (static_cast<uint16_t>(*siz0) << 8) | *siz1;
        }
        
        /// the marker for EXIF byte regions:
        const std::array<byte, 2> marker{ 0xFF, 0xE1 };
    
    } /// namespace (anon.)
    
    std::unique_ptr<Image> JPEGFormat::read(byte_source* src,
                                            ImageFactory* factory,
                                            options_map const& opts)  {
        
        jpeg_source_adaptor adaptor(src);
        jpeg_decompress_holder decompressor;
        
        /// error management
        error_mgr jerr;
        decompressor.info.err = &jerr.pub;
        
        /// source
        decompressor.info.src = &adaptor.mgr;
        
        /// now read the header & image data
        decompressor.start();
        
        if (jerr.has_error()) {
            imread_raise(CannotReadError,
                "libjpeg internal error:",
                jerr.error_message);
        }
        
        const int h = decompressor.height();
        const int w = decompressor.width();
        const int d = decompressor.components();
        
        std::unique_ptr<Image> output(factory->create(8, h, w, d));
        
        /// allocate single-row sample array
        JSAMPARRAY samples = decompressor.allocate_samples(w, d, 1);
        
        /// Hardcoding uint8_t as the type for now
        int c_stride = (d == 1) ? 0 : output->stride(2);
        uint8_t* __restrict__ ptr = output->rowp_as<uint8_t>(0);
        
        while (decompressor.scanline() < decompressor.height()) {
            decompressor.read_samples(samples);
            JSAMPLE* srcPtr = samples[0];
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < d; ++c) {
                    /// theoretically you would want to scale this next bit,
                    /// depending on whatever the bit depth may be
                    /// -- SOMEDAAAAAAAAAAAAAAY.....
                    pix::convert(*srcPtr++, ptr[c*c_stride]);
                }
                ++ptr;
            }
        }
        
        if (jerr.has_error()) {
            imread_raise(CannotReadError,
                "libjpeg internal error:",
                jerr.error_message);
        }
        
        return output;
    }
    
    
    options_map JPEGFormat::read_metadata(byte_source* src,
                                          options_map const& opts) {
        using easyexif::EXIFInfo;
        using im::byte_iterator;
        
        options_map out;
        // src->seek_absolute(0);
        
        byte_iterator result = std::search(src->begin(),   src->end(),
                                           marker.begin(), marker.end());
        bool has_exif = result != src->end();
        if (!has_exif) { return out; }
        
        bytevec_t data;
        uint16_t size = parse_size(result);
        data.reserve(size);
        
        std::advance(result, 4);
        std::copy(result, result + size,
                  std::back_inserter(data));
        
        EXIFInfo exif;
        if (exif.parseFromEXIFSegment(&data[0], data.size()) != PARSE_EXIF_SUCCESS) {
            imread_raise(MetadataReadError,
                "Error parsing JPEG EXIF metadata");
            // return out;
        }
        
        /// strings
        out.set("ImageDescription",     exif.ImageDescription);
        out.set("Make",                 exif.Make);
        out.set("Model",                exif.Model);
        out.set("Software",             exif.Software);
        out.set("DateTime",             exif.DateTime);
        out.set("DateTimeOriginal",     exif.DateTimeOriginal);
        out.set("DateTimeDigitized",    exif.DateTimeDigitized);
        out.set("SubSecTimeOriginal",   exif.SubSecTimeOriginal);
        out.set("Copyright",            exif.Copyright);
        
        /// numbers
        out.set("BitsPerSample",        exif.BitsPerSample);
        out.set("ExosureTime",          exif.ExposureTime);
        out.set("FNumber",              exif.FNumber);
        out.set("ISOSpeedRatings",      exif.ISOSpeedRatings);
        out.set("ShutterSpeedValue",    exif.ShutterSpeedValue);
        out.set("ExposureBiasValue",    exif.ExposureBiasValue);
        out.set("SubjectDistance",      exif.SubjectDistance);
        out.set("FocalLength",          exif.FocalLength);
        out.set("FocalLengthIn35mm",    exif.FocalLengthIn35mm);
        out.set("ImageWidth",           exif.ImageWidth);
        out.set("ImageHeight",          exif.ImageHeight);
        
        /// “enums”
        out.set("Orientation",          exif.Orientation);       /// 0: unspecified in EXIF data
                                                                 /// 1: upper left of image
                                                                 /// 3: lower right of image
                                                                 /// 6: upper right of image
                                                                 /// 8: lower left of image
                                                                 /// 9: undefined
        out.set("Flash",                exif.Flash);             /// 0: no flash, 1: flash
        out.set("MeteringMode",         exif.MeteringMode);      /// 1: average
                                                                 /// 2: center weighted average
                                                                 /// 3: spot
                                                                 /// 4: multi-spot
                                                                 /// 5: multi-segment
        
        /// return the options_map full of metadata:
        return out;
    }
    
    void JPEGFormat::write(Image& input,
                           byte_sink* output,
                           options_map const& opts) {
        
        /// sanity-check input bit depth
        if (input.nbits() != 8) {
            imread_raise(CannotWriteError,
                FF("Image must be 8 bits for JPEG saving (got %i)",
                    input.nbits()));
        }
        
        jpeg_dst_adaptor adaptor(output);
        jpeg_compress_holder compressor;
        
        /// error management
        error_mgr jerr;
        compressor.info.err = &jerr.pub;
        
        /// destination
        compressor.info.dest = &adaptor.mgr;
        
        const int w = input.dim(0);
        const int h = input.dim(1);
        const int d = std::min(3, input.dim(2));
        
        /// assign image values
        compressor.set_width(w);
        compressor.set_height(h);
        compressor.set_components(d);
        
        /// set 'defaults'
        compressor.set_defaults();
        
        /// set quality value (0 ≤ quality ≤ 100)
        compressor.set_quality(opts.cast<std::size_t>("jpg:quality", 100));
        
        /// start compression!
        compressor.start();
        
        /// error check
        if (jerr.has_error()) {
            imread_raise(CannotWriteError,
                "libjpeg internal error:",
                jerr.error_message);
        }
        
        /// allocate row buffer
        JSAMPLE* rowbuf = new JSAMPLE[w * d]; /// width * channels
        
        /// access pixels as type JSAMPLE
        pix::accessor<JSAMPLE> at = input.access<JSAMPLE>();
        
        /// write scanlines in pixel loop
        while (compressor.next_scanline() < compressor.height()) {
            JSAMPLE* __restrict__ dstPtr = rowbuf;
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < d; ++c) {
                    pix::convert(
                        at(x, compressor.next_scanline(), c)[0],
                        *dstPtr++);
                }
            }
            compressor.write_scanlines(&rowbuf);
        }
        
        /// error check
        if (jerr.has_error()) {
            imread_raise(CannotWriteError,
                "libjpeg internal error:",
                jerr.error_message);
        }
        
        /// deallocate row buffer
        delete[] rowbuf;
    }
}
