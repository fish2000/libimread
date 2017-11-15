/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <csetjmp>
#include <algorithm>
#include <vector>
#include <memory>
#include <array>

#include <iod/json.hh>

#include <libimread/errors.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>
#include <libimread/metadata.hh>
#include <libimread/pixels.hh>
#include <libimread/ext/exif.hh>

extern "C" {
#include <jpeglib.h>
}

#define BOOLEAN_TRUE()  static_cast<boolean>(true)
#define BOOLEAN_FALSE() static_cast<boolean>(false)

namespace im {
    
    DECLARE_FORMAT_OPTIONS(JPEGFormat);
    
    namespace {
        
        const std::size_t buffer_size = JPEGFormat::options.buffer_size;
        using byte_ptr = std::unique_ptr<byte[]>;
        
        /// forward-declare this one non-NOP function
        /// whose pointer is referenced more than once:
        boolean fill_input_buffer(j_decompress_ptr cinfo);
        
        /// Adaptor-specific NOP functions:
        void NOP_SRC(j_decompress_ptr)  {}
        void NOP_DST(j_compress_ptr)    {}
        
        /// RAII-ish wrapper for holding the `jpeg_source_mgr` structure,
        /// initializing it with functions binding it to an instance of
        /// `im::byte_source` that, in turn, reads from an underlying
        /// byte array (both of which are allocated as member instances).
        struct JPEGSourceAdaptor {
            jpeg_source_mgr mgr;
            byte_source* source;
            byte_ptr buffer;
            
            JPEGSourceAdaptor(byte_source* s)
                :source(s)
                ,buffer{ std::make_unique<byte[]>(buffer_size) }
                {
                    mgr.next_input_byte     = buffer.get();
                    mgr.bytes_in_buffer     = 0;
                    mgr.init_source         = NOP_SRC;
                    mgr.fill_input_buffer   = fill_input_buffer;
                    
                    mgr.skip_input_data     = [](j_decompress_ptr cinfo, long num_bytes) {
                        if (num_bytes <= 0) { return; }
                        JPEGSourceAdaptor* adaptor = reinterpret_cast<JPEGSourceAdaptor*>(cinfo->src);
                        while (num_bytes > long(adaptor->mgr.bytes_in_buffer)) {
                            num_bytes -= adaptor->mgr.bytes_in_buffer;
                            fill_input_buffer(cinfo);
                        }
                        adaptor->mgr.next_input_byte += num_bytes;
                        adaptor->mgr.bytes_in_buffer -= num_bytes;
                    };
                    
                    mgr.resync_to_restart   = jpeg_resync_to_restart;
                    mgr.term_source         = NOP_SRC;
                }
            
            JPEGSourceAdaptor(JPEGSourceAdaptor const&) = delete;
            JPEGSourceAdaptor(JPEGSourceAdaptor&&) = delete;
        };
        
        /// RAII-ish wrapper for holding the `jpeg_destination_mgr` structure,
        /// initializing it with functions binding it to an instance of
        /// `im::byte_sink` that, in turn, writes to an underlying
        /// byte array (both of which are allocated as member instances).
        struct JPEGDestinationAdaptor {
            jpeg_destination_mgr mgr;
            byte_sink* sink;
            byte_ptr buffer;
            
            JPEGDestinationAdaptor(byte_sink* s)
                :sink(s)
                ,buffer{ std::make_unique<byte[]>(buffer_size) }
                {
                    mgr.next_output_byte    = buffer.get();
                    mgr.free_in_buffer      = buffer_size;
                    mgr.init_destination    = NOP_DST;
                    
                    mgr.empty_output_buffer = [](j_compress_ptr cinfo) {
                        JPEGDestinationAdaptor* adaptor = reinterpret_cast<JPEGDestinationAdaptor*>(cinfo->dest);
                        adaptor->sink->write(adaptor->buffer.get(),
                                             buffer_size);
                        adaptor->mgr.next_output_byte = adaptor->buffer.get();
                        adaptor->mgr.free_in_buffer = buffer_size;
                        return BOOLEAN_TRUE();
                    };
                    
                    mgr.term_destination    = [](j_compress_ptr cinfo) {
                        JPEGDestinationAdaptor* adaptor = reinterpret_cast<JPEGDestinationAdaptor*>(cinfo->dest);
                        adaptor->sink->write(adaptor->buffer.get(),
                                             adaptor->mgr.next_output_byte - adaptor->buffer.get());
                        adaptor->sink->flush();
                    };
                }
            
            JPEGDestinationAdaptor(JPEGDestinationAdaptor const&) = delete;
            JPEGDestinationAdaptor(JPEGDestinationAdaptor&&) = delete;
        };
        
        /// definition for `fill_input_buffer(…)` as declared above:
        boolean fill_input_buffer(j_decompress_ptr cinfo) {
            JPEGSourceAdaptor* adaptor = reinterpret_cast<JPEGSourceAdaptor*>(cinfo->src);
            adaptor->mgr.next_input_byte = adaptor->buffer.get();
            adaptor->mgr.bytes_in_buffer = adaptor->source->read(adaptor->buffer.get(), buffer_size);
            return BOOLEAN_TRUE();
        }
        
        /// function to match a J_COLOR_SPACE constant up (coarsely) to
        /// an image, per the number of color channels it has:
        inline J_COLOR_SPACE color_space(int components) {
            switch (components) {
                case 3: return JCS_RGB;
                case 1: return JCS_GRAYSCALE;
                case 4: return JCS_CMYK;
                default: {
                    imread_raise(CannotReadError,
                        "\tim::(anon)::color_space() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
                     FF("\tim::(anon)::color_space() got:    `components` = (int){ %i }", components),
                        "\tim::(anon)::color_space() needs:  `components` = (int){ 1, 3, 4 }");
                }
            }
        }
        
        /// Base class for JPEG compressor wrappers:
        /// holds an instance of the `jpeg_error_mgr` structure,
        /// wrapped in a very simple inner-holding class,
        /// which furnishes some accessors and a `jmp_buf` setup,
        /// used by jpeglib for raising exception-style conditions:
        struct JPEGCompressionBase {
            
            public:
                struct ErrorManager {
                    
                    ErrorManager() {
                        jpeg_std_error(&pub);
                        pub.error_exit = [](j_common_ptr cinfo) {
                            using ErrorManager = JPEGCompressionBase::ErrorManager;
                            ErrorManager* err = reinterpret_cast<ErrorManager*>(cinfo->err);
                            (*cinfo->err->format_message)(cinfo, err->error_message);
                            longjmp(err->setjmp_buffer, 1);
                        };
                        error_message[0] = 0;
                    }
                    
                    mutable struct jpeg_error_mgr pub;
                    mutable jmp_buf setjmp_buffer;
                    mutable char error_message[JMSG_LENGTH_MAX];
                    
                } errormgr;
            
            public:
                bool has_error() const {
                    return setjmp(errormgr.setjmp_buffer);
                }
                
                std::string error_message() const {
                    return std::string(errormgr.error_message);
                }
            
        };
        
        /// JPEG decompressor API class, wrapping the `jpeg_decompress_struct`
        /// from jpeglib, alongside a JPEGSourceAdapter instance (q.v. implementation,
        /// above) and furnishing accessor shortcut methods.
        struct JPEGDecompressor : public JPEGCompressionBase {
            
            JPEGDecompressor(byte_source* source)
                :adaptor(source)
                {
                    jpeg_create_decompress(&info);
                    info.err = &errormgr.pub;
                    info.src = &adaptor.mgr;
                }
            
            ~JPEGDecompressor() {
                jpeg_finish_decompress(&info);
                jpeg_destroy_decompress(&info);
            }
            
            void start() {
                jpeg_read_header(&info, BOOLEAN_TRUE());
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
                JPEGSourceAdaptor adaptor;
                jpeg_decompress_struct info;
        };
        
        /// JPEG compressor API class, wrapping the `jpeg_compress_struct`
        /// from jpeglib, alongside a JPEGDestinationAdapter instance (q.v. implementation,
        /// above) and furnishing accessor shortcut methods.
        struct JPEGCompressor : public JPEGCompressionBase {
            
            JPEGCompressor(byte_sink* sink)
                :adaptor(sink)
                {
                    jpeg_create_compress(&info);
                    info.err = &errormgr.pub;
                    info.dest = &adaptor.mgr;
                }
            
            ~JPEGCompressor() {
                jpeg_finish_compress(&info);
                jpeg_destroy_compress(&info);
            }
            
            void start() {
                jpeg_start_compress(&info, BOOLEAN_TRUE());
            }
            
            void set_defaults() {
                jpeg_set_defaults(&info);
            }
            
            void set_quality(std::size_t quality) {
                if (quality > 100) { quality = 100; }
                jpeg_set_quality(&info, quality, BOOLEAN_FALSE());
            }
            
            int height() const          { return info.image_height; }
            int width() const           { return info.image_width; }
            int components() const      { return info.input_components; }
            int next_scanline() const   { return info.next_scanline; }
            
            void set_height(int height)         { info.image_height = height; }
            void set_width(int width)           { info.image_width = width; }
            void set_components(int components) { info.input_components = components;
                                                  info.in_color_space = color_space(components); }
            
            void write_scanlines(JSAMPLE** rows, int idx = 1) {
                jpeg_write_scanlines(&info, rows, idx);
            }
            
            public:
                JPEGDestinationAdaptor adaptor;
                jpeg_compress_struct info;
        };
        
        /// Shortcut template function to extract the size
        /// of an EXIF byte region:
        template <typename Iterator>
        uint16_t parse_size(Iterator it) {
            Iterator siz0 = std::next(it, 2);
            Iterator siz1 = std::next(it, 3);
            return (static_cast<uint16_t>(*siz0) << 8) | *siz1;
        }
        
        /// The byte marker signature for EXIF byte regions:
        const std::array<byte, 2> marker{ 0xFF, 0xE1 };
    
    } /// namespace (anon.)
    
    std::unique_ptr<Image> JPEGFormat::read(byte_source* src,
                                            ImageFactory* factory,
                                            Options const& opts)  {
        
        JPEGDecompressor decompressor(src);
        
        /// first, read the header & image data:
        decompressor.start();
        
        /// initial error check:
        if (decompressor.has_error()) {
            imread_raise(CannotReadError,
                "libjpeg internal error:",
                decompressor.error_message());
        }
        
        /// stash dimension values:
        const int h = decompressor.height();
        const int w = decompressor.width();
        const int d = decompressor.components();
        
        /// create the output image:
        std::unique_ptr<Image> output(factory->create(8, h, w, d));
        
        /// allocate a single-row sample array:
        JSAMPARRAY samples = decompressor.allocate_samples(w, d, 1);
        
        /// Hardcoding uint8_t as the type for now:
        int c_stride = (d == 1) ? 0 : output->stride(2);
        uint8_t* __restrict__ ptr = output->rowp_as<uint8_t>(0);
        
        /// read scanlines in loop:
        while (decompressor.scanline() < h) {
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
        
        /// final error check:
        if (decompressor.has_error()) {
            imread_raise(CannotReadError,
                "libjpeg internal error:",
                decompressor.error_message());
        }
        
        return output;
    }
    
    Metadata JPEGFormat::read_metadata(byte_source* src,
                                       Options const& opts) {
        using easyexif::EXIFInfo;
        using im::byte_iterator;
        
        Metadata meta;
        // src->seek_absolute(0);
        
        byte_iterator result = std::search(src->begin(),   src->end(),
                                           marker.begin(), marker.end());
        bool has_exif = result != src->end();
        if (!has_exif) { return meta; }
        
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
            // return meta;
        }
        
        /// strings
        meta.set("ImageDescription",     exif.ImageDescription);
        meta.set("Make",                 exif.Make);
        meta.set("Model",                exif.Model);
        meta.set("Software",             exif.Software);
        meta.set("DateTime",             exif.DateTime);
        meta.set("DateTimeOriginal",     exif.DateTimeOriginal);
        meta.set("DateTimeDigitized",    exif.DateTimeDigitized);
        meta.set("SubSecTimeOriginal",   exif.SubSecTimeOriginal);
        meta.set("Copyright",            exif.Copyright);
        
        /// numbers
        meta.set("BitsPerSample",        std::to_string(exif.BitsPerSample));
        meta.set("ExosureTime",          std::to_string(exif.ExposureTime));
        meta.set("FNumber",              std::to_string(exif.FNumber));
        meta.set("ISOSpeedRatings",      std::to_string(exif.ISOSpeedRatings));
        meta.set("ShutterSpeedValue",    std::to_string(exif.ShutterSpeedValue));
        meta.set("ExposureBiasValue",    std::to_string(exif.ExposureBiasValue));
        meta.set("SubjectDistance",      std::to_string(exif.SubjectDistance));
        meta.set("FocalLength",          std::to_string(exif.FocalLength));
        meta.set("FocalLengthIn35mm",    std::to_string(exif.FocalLengthIn35mm));
        meta.set("ImageWidth",           std::to_string(exif.ImageWidth));
        meta.set("ImageHeight",          std::to_string(exif.ImageHeight));
        
        /// “enums”
        meta.set("Orientation",          std::to_string(exif.Orientation));      /// 0: unspecified in EXIF data
                                                                 /// 1: upper left of image
                                                                 /// 3: lower right of image
                                                                 /// 6: upper right of image
                                                                 /// 8: lower left of image
                                                                 /// 9: undefined
        meta.set("Flash",                std::to_string(exif.Flash));            /// 0: no flash, 1: flash
        meta.set("MeteringMode",         std::to_string(exif.MeteringMode));     /// 1: average
                                                                 /// 2: center weighted average
                                                                 /// 3: spot
                                                                 /// 4: multi-spot
                                                                 /// 5: multi-segment
        
        /// return the metadata instance:
        return meta;
    }
    
    void JPEGFormat::write(Image& input,
                           byte_sink* output,
                           Options const& opts) {
        
        /// sanity-check input bit depth
        if (input.nbits() != 8) {
            imread_raise(CannotWriteError,
                FF("Image must be 8 bits for JPEG saving (got %i)",
                    input.nbits()));
        }
        
        JPEGCompressor compressor(output);
        
        const int w = input.dim(0);
        const int h = input.dim(1);
        const int d = std::min(3, input.dim(2));
        
        /// assign image values:
        compressor.set_width(w);
        compressor.set_height(h);
        compressor.set_components(d);
        
        /// set 'defaults':
        compressor.set_defaults();
        
        /// set quality value (0 ≤ quality ≤ 100):
        compressor.set_quality(opts.cast<std::size_t>("jpg:quality", 100));
        
        /// start the compression!
        compressor.start();
        
        /// first error check:
        if (compressor.has_error()) {
            imread_raise(CannotWriteError,
                "libjpeg internal error:",
                compressor.error_message());
        }
        
        /// allocate a row buffer:
        std::unique_ptr<JSAMPLE[]> rowbuf = std::make_unique<JSAMPLE[]>(w * d);
        JSAMPLE* rowptr = rowbuf.get();
        
        /// access pixels as type JSAMPLE:
        pix::accessor<JSAMPLE> at = input.access<JSAMPLE>();
        
        /// write scanlines in pixel loop:
        while (compressor.next_scanline() < h) {
            JSAMPLE* __restrict__ dstPtr = rowbuf.get();
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < d; ++c) {
                    pix::convert(
                        at(x, compressor.next_scanline(), c)[0],
                        *dstPtr++);
                }
            }
            compressor.write_scanlines(std::addressof(rowptr));
        }
        
        /// final error check:
        if (compressor.has_error()) {
            imread_raise(CannotWriteError,
                "libjpeg internal error:",
                compressor.error_message());
        }
        
    }
}
