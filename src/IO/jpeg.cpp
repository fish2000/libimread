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
#include <libimread/ext/exif.hh>

extern "C" {
#include <jpeglib.h>
}

/// “boolean” is a jpeglib type, evidently:
#define BOOLEAN_TRUE()  (static_cast<boolean>(true))
#define BOOLEAN_FALSE() (static_cast<boolean>(false))

namespace im {
    
    DECLARE_FORMAT_OPTIONS(JPEGFormat);
    
    namespace {
        
        using byte_ptr = std::unique_ptr<byte[]>;
        
        /// Constants
        const std::size_t kBufferSize = JPEGFormat::options.buffer_size;
        const std::size_t kDefaultQuality = static_cast<std::size_t>(JPEGFormat::options.writeopts.quality * 100);
        
        /// Adaptor-specific NOP function types:
        using NOP_SRC = std::add_pointer_t<void(j_decompress_ptr)>;
        using NOP_DST = std::add_pointer_t<void(j_compress_ptr)>;
        
        /// RAII-ish wrapper for holding the `jpeg_source_mgr` structure,
        /// initializing it with functions binding it to an instance of
        /// `im::byte_source` that, in turn, reads from an underlying
        /// byte array (both of which are allocated as member instances).
        struct JPEGSourceAdaptor {
            constexpr static const NOP_SRC NOP = [](j_decompress_ptr) -> void {};
            
            jpeg_source_mgr mgr;
            byte_source* source;
            byte_ptr buffer;
            
            JPEGSourceAdaptor(byte_source* s)
                :source(s)
                ,buffer{ std::make_unique<byte[]>(kBufferSize) }
                {
                    mgr.next_input_byte     = buffer.get();
                    mgr.bytes_in_buffer     = 0;
                    mgr.init_source         = NOP;
                    mgr.fill_input_buffer   = [](j_decompress_ptr cinfo) -> boolean {
                        JPEGSourceAdaptor* adaptor = reinterpret_cast<JPEGSourceAdaptor*>(cinfo->src);
                        jpeg_source_mgr& mgr = adaptor->mgr;
                        mgr.next_input_byte = adaptor->buffer.get();
                        mgr.bytes_in_buffer = adaptor->source->read(adaptor->buffer.get(),
                                                                             kBufferSize);
                        return BOOLEAN_TRUE();
                    };
                    
                    mgr.skip_input_data     = [](j_decompress_ptr cinfo, long num_bytes) -> void {
                        if (num_bytes <= 0) { return; }
                        jpeg_source_mgr& mgr = reinterpret_cast<JPEGSourceAdaptor*>(cinfo->src)->mgr;
                        while (num_bytes > long(mgr.bytes_in_buffer)) {
                            num_bytes -= mgr.bytes_in_buffer;
                            mgr.fill_input_buffer(cinfo); /// calls lambda defined above
                        }
                        mgr.next_input_byte += num_bytes;
                        mgr.bytes_in_buffer -= num_bytes;
                    };
                    
                    mgr.resync_to_restart   = jpeg_resync_to_restart;
                    mgr.term_source         = NOP;
                }
            
            JPEGSourceAdaptor(JPEGSourceAdaptor const&) = delete;
            JPEGSourceAdaptor(JPEGSourceAdaptor&&) = delete;
        };
        
        /// RAII-ish wrapper for holding the `jpeg_destination_mgr` structure,
        /// initializing it with functions binding it to an instance of
        /// `im::byte_sink` that, in turn, writes to an underlying
        /// byte array (both of which are allocated as member instances).
        struct JPEGDestinationAdaptor {
            constexpr static const NOP_DST NOP = [](j_compress_ptr) -> void {};
            
            jpeg_destination_mgr mgr;
            byte_sink* sink;
            byte_ptr buffer;
            
            JPEGDestinationAdaptor(byte_sink* s)
                :sink(s)
                ,buffer{ std::make_unique<byte[]>(kBufferSize) }
                {
                    mgr.next_output_byte    = buffer.get();
                    mgr.free_in_buffer      = kBufferSize;
                    mgr.init_destination    = NOP;
                    
                    mgr.empty_output_buffer = [](j_compress_ptr cinfo) -> boolean {
                        JPEGDestinationAdaptor* adaptor = reinterpret_cast<JPEGDestinationAdaptor*>(cinfo->dest);
                        jpeg_destination_mgr& mgr = adaptor->mgr;
                        adaptor->sink->write(adaptor->buffer.get(),
                                             kBufferSize);
                        mgr.next_output_byte = adaptor->buffer.get();
                        mgr.free_in_buffer = kBufferSize;
                        return BOOLEAN_TRUE();
                    };
                    
                    mgr.term_destination    = [](j_compress_ptr cinfo) -> void {
                        JPEGDestinationAdaptor* adaptor = reinterpret_cast<JPEGDestinationAdaptor*>(cinfo->dest);
                        adaptor->sink->write(adaptor->buffer.get(),
                                             adaptor->mgr.next_output_byte - adaptor->buffer.get());
                        adaptor->sink->flush();
                    };
                }
            
            JPEGDestinationAdaptor(JPEGDestinationAdaptor const&) = delete;
            JPEGDestinationAdaptor(JPEGDestinationAdaptor&&) = delete;
        };
        
        /// function to match a J_COLOR_SPACE constant up (coarsely) to
        /// an image, per the number of color channels it has:
        inline J_COLOR_SPACE color_space_for_components(int components) {
            switch (components) {
                case 3: return JCS_RGB;
                case 1: return JCS_GRAYSCALE;
                case 4: return JCS_CMYK;
                default: {
                    imread_raise(JPEGIOError,
                        "\tim::(anon)::color_space_for_components() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
                     FF("\tim::(anon)::color_space_for_components() got:    `components` = (int){ %i }", components),
                        "\tim::(anon)::color_space_for_components() needs:  `components` = (int){ 1, 3, 4 }");
                }
            }
        }
        
        /// ... basically, the inverse of the 
        inline int components_for_color_space(J_COLOR_SPACE color_space) {
            switch (color_space) {
                case JCS_RGB: return 3;
                case JCS_GRAYSCALE: return 1;
                case JCS_CMYK: return 4;
                default: {
                    imread_raise(JPEGIOError,
                        "\tim::(anon)::components_for_color_space() says:   \"UNSUPPORTED COLOR SPACE CONSTANT\"",
                     FF("\tim::(anon)::components_for_color_space() got:    `color_space` = (J_COLOR_SPACE){ %i }", color_space),
                        "\tim::(anon)::components_for_color_space() needs:  `color_space` = (J_COLOR_SPACE)JCS_{ RGB, GRAYSCALE, CMYK }");
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
                using jpeg_error_mgr_t = struct jpeg_error_mgr;
            
            public:
                struct ErrorManager {
                    mutable jpeg_error_mgr_t mgr;
                    mutable jmp_buf jumpbuffer;
                    mutable char message[JMSG_LENGTH_MAX];
                    
                    ErrorManager() {
                        jpeg_std_error(&mgr);
                        mgr.error_exit = [](j_common_ptr cinfo) {
                            using ErrorManager = JPEGCompressionBase::ErrorManager;
                            ErrorManager* err = reinterpret_cast<ErrorManager*>(cinfo->err);
                            (*cinfo->err->format_message)(cinfo, err->message);
                            longjmp(err->jumpbuffer, 1);
                        };
                        message[0] = 0;
                    }
                    
                } error;
            
            public:
                virtual ~JPEGCompressionBase() {}
                
                bool has_error() const {
                    return setjmp(error.jumpbuffer);
                }
                
                std::string error_message() const {
                    return std::string(error.message);
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
                    info.err = &error.mgr;
                    info.src = &adaptor.mgr;
                }
            
            virtual ~JPEGDecompressor() {
                jpeg_finish_decompress(&info);
                jpeg_destroy_decompress(&info);
            }
            
            void start() {
                jpeg_read_header(&info, BOOLEAN_TRUE());
                jpeg_start_decompress(&info);
            }
            
            int height() const                  { return info.output_height; }
            int width() const                   { return info.output_width; }
            J_COLOR_SPACE color_space() const   { return color_space_for_components(info.output_components); }
            int components() const              { return info.output_components; }
            int scanline() const                { return info.output_scanline; }
            
            JSAMPARRAY allocate_samples(int components = 0,
                                        int width = 0,
                                        int height = 1) {
                /// blow up for invalid heights:
                if (height < 1) {
                    imread_raise(JPEGIOError,
                       "\tim::(anon)::JPEGDecompressor::allocate_samples() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
                    FF("\tim::(anon)::JPEGDecompressor::allocate_samples() got:    `height` = (int){ %i }", height),
                       "\tim::(anon)::JPEGDecompressor::allocate_samples() needs:  `height` > (int){ 0 }");
                }
                /// default to values read from JPEG header:
                if (!width)      { width = info.output_width;      }
                if (!components) { components = info.output_components; }
                /// allocate using internal function pointer:
                return (*info.mem->alloc_sarray)(reinterpret_cast<j_common_ptr>(&info),
                                                                  JPOOL_IMAGE,
                                                                  width * components,
                                                                  height);
            }
            
            JSAMPARRAY allocate_samples(J_COLOR_SPACE color_space,
                                        int width = 0,
                                        int height = 1) {
                /// convert the color space constant,
                /// and delegate to the all-ints version above:
                return allocate_samples(components_for_color_space(color_space),
                                        width,
                                        height);
            }
            
            void read_scanlines(JSAMPLE** rows, int idx = 1) {
                jpeg_read_scanlines(&info, rows, idx);
            }
            
            void read_scanlines(JSAMPLE* rows, int idx = 1) {
                jpeg_read_scanlines(&info, std::addressof(rows), idx);
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
                    info.err = &error.mgr;
                    info.dest = &adaptor.mgr;
                }
            
            virtual ~JPEGCompressor() {
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
            
            int height() const                  { return info.image_height; }
            int width() const                   { return info.image_width; }
            J_COLOR_SPACE color_space() const   { return color_space_for_components(info.input_components); }
            int components() const              { return info.input_components; }
            int next_scanline() const           { return info.next_scanline; }
            
            void set_height(int height)         { info.image_height = height; }
            void set_width(int width)           { info.image_width = width; }
            void set_color_space(J_COLOR_SPACE color_space) {
                                                  info.input_components = components_for_color_space(color_space);
                                                  info.in_color_space = color_space; }
            void set_components(int components) { info.input_components = components;
                                                  info.in_color_space = color_space_for_components(components); }
            
            JSAMPARRAY allocate_samples(int components = 0,
                                        int width = 0,
                                        int height = 1) {
                /// blow up for invalid heights:
                if (height < 1) {
                    imread_raise(JPEGIOError,
                       "\tim::(anon)::JPEGCompressor::allocate_samples() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
                    FF("\tim::(anon)::JPEGCompressor::allocate_samples() got:    `height` = (int){ %i }", height),
                       "\tim::(anon)::JPEGCompressor::allocate_samples() needs:  `height` > (int){ 0 }");
                }
                /// default to values attached to ‘info’ struct:
                if (!width)      { width = info.image_width;      }
                if (!components) { components = info.input_components; }
                /// allocate using internal function pointer:
                return (*info.mem->alloc_sarray)(reinterpret_cast<j_common_ptr>(&info),
                                                                  JPOOL_IMAGE,
                                                                  width * components,
                                                                  height);
            }
            
            JSAMPARRAY allocate_samples(J_COLOR_SPACE color_space,
                                        int width = 0,
                                        int height = 1) {
                /// convert the color space constant,
                /// and delegate to the all-ints version above:
                return allocate_samples(components_for_color_space(color_space),
                                        width,
                                        height);
            }
            
            void write_scanlines(JSAMPLE** rows, int idx = 1) {
                jpeg_write_scanlines(&info, rows, idx);
            }
            
            void write_scanlines(JSAMPLE* rows, int idx = 1) {
                jpeg_write_scanlines(&info, std::addressof(rows), idx);
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
            imread_raise(JPEGIOError,
                "libjpeg internal error:",
                decompressor.error_message());
        }
        
        /// stash dimension values:
        const int h = decompressor.height();
        const int w = decompressor.width();
        const int c = decompressor.components();
        
        /// create the output image:
        std::unique_ptr<Image> output(factory->create(8, h, w, c));
        
        /// allocate a single-row sample array:
        JSAMPARRAY samples = decompressor.allocate_samples();
        
        /// Hardcoding JSAMPLE (== uint8_t) as the type for now:
        int color_stride = (c == 1) ? 0 : output->stride(2);
        JSAMPLE* __restrict__ ptr = output->rowp_as<JSAMPLE>(0);
        
        /// read scanlines in loop:
        while (decompressor.scanline() < h) {
            decompressor.read_scanlines(samples);
            JSAMPLE* __restrict__ srcPtr = samples[0];
            for (int x = 0; x < w; ++x) {
                for (int cc = 0; cc < c; ++cc) {
                    /// theoretically you would want to scale this next value,
                    /// depending on the bit depth -- SOMEDAAAAAAAAAAAAAAY.....
                    ptr[cc*color_stride] = *srcPtr++;
                }
                ++ptr;
            }
        }
        
        /// final error check:
        if (decompressor.has_error()) {
            imread_raise(JPEGIOError,
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
        const int c = std::min(3, input.dim(2));
        
        /// assign image values:
        compressor.set_width(w);
        compressor.set_height(h);
        compressor.set_components(c);
        
        /// set 'defaults':
        compressor.set_defaults();
        
        /// set quality value (0 ≤ quality ≤ 100):
        compressor.set_quality(
              opts.cast<std::size_t>("jpg:quality",
                                      kDefaultQuality));
        
        /// start the compression!
        compressor.start();
        
        /// initial error check:
        if (compressor.has_error()) {
            imread_raise(JPEGIOError,
                "libjpeg internal error:",
                compressor.error_message());
        }
        
        /// allocate a single-row sample array:
        JSAMPARRAY samples = compressor.allocate_samples();
        
        /// view data as type JSAMPLE:
        av::strided_array_view<JSAMPLE, 3> view = input.view<JSAMPLE>();
        
        /// write scanlines in pixel loop:
        int scan;
        while ((scan = compressor.next_scanline()) < h) {
            JSAMPLE* __restrict__ dstPtr = samples[0];
            for (int x = 0; x < w; ++x) {
                for (int cc = 0; cc < c; ++cc) {
                    *dstPtr++ = view[{x, scan, cc}];
                }
            }
            compressor.write_scanlines(samples);
        }
        
        /// final error check:
        if (compressor.has_error()) {
            imread_raise(JPEGIOError,
                "libjpeg internal error:",
                compressor.error_message());
        }
        
    }
}
