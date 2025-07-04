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
    #include "iccjpeg/iccjpeg.h"
}

/// “boolean” is a jpeglib type, evidently:
#define BOOLEAN_TRUE()  (static_cast<boolean>(true))
#define BOOLEAN_FALSE() (static_cast<boolean>(false))

namespace im {
    
    DECLARE_FORMAT_OPTIONS(JPEGFormat);
    
    namespace {
        
        /// Constants
        const bool        kDefaultProgressive = static_cast<bool>(JPEGFormat::options.writeopts.progressive);
        const std::size_t kDefaultQuantization = JPEGFormat::options.quantization;
        const std::size_t kDefaultQuality = static_cast<std::size_t>(JPEGFormat::options.writeopts.quality * 100);
        const std::size_t kBufferSize = JPEGFormat::options.buffer_size;
        
        /// Unique pointer type, holding an array of JPEG bytes:
        using byte_ptr = std::unique_ptr<JOCTET[]>;
        
        /// Adaptor-specific NOP function types:
        using nop_src_t = std::add_pointer_t<void(j_decompress_ptr)>;
        using nop_dst_t = std::add_pointer_t<void(j_compress_ptr)>;
        
        /// RAII-ish wrapper for holding the `jpeg_source_mgr` structure,
        /// initializing it with functions binding it to an instance of
        /// `im::byte_source` that, in turn, reads from an underlying
        /// byte array (both of which are allocated as member instances).
        struct JPEGSourceAdaptor {
            constexpr static const nop_src_t NOP = [](j_decompress_ptr) -> void {};
            
            jpeg_source_mgr mgr;
            byte_source* source;
            byte_ptr buffer;
            
            JPEGSourceAdaptor(byte_source* s)
                :source(s)
                ,buffer{ std::make_unique<JOCTET[]>(kBufferSize) }
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
            
            JPEGSourceAdaptor(void) = delete;
            JPEGSourceAdaptor(JPEGSourceAdaptor const&) = delete;
            JPEGSourceAdaptor(JPEGSourceAdaptor&&) = delete;
        };
        
        /// RAII-ish wrapper for holding the `jpeg_destination_mgr` structure,
        /// initializing it with functions binding it to an instance of
        /// `im::byte_sink` that, in turn, writes to an underlying
        /// byte array (both of which are allocated as member instances).
        struct JPEGDestinationAdaptor {
            constexpr static const nop_dst_t NOP = [](j_compress_ptr) -> void {};
            
            jpeg_destination_mgr mgr;
            byte_sink* sink;
            byte_ptr buffer;
            
            JPEGDestinationAdaptor(byte_sink* s)
                :sink(s)
                ,buffer{ std::make_unique<JOCTET[]>(kBufferSize) }
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
            
            JPEGDestinationAdaptor(void) = delete;
            JPEGDestinationAdaptor(JPEGDestinationAdaptor const&) = delete;
            JPEGDestinationAdaptor(JPEGDestinationAdaptor&&) = delete;
        };
        
        /// function to match a J_COLOR_SPACE constant up (coarsely) to
        /// an image, per the number of color channels it has:
        static J_COLOR_SPACE color_space_for_components(int components) {
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
        static int components_for_color_space(J_COLOR_SPACE color_space) {
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
                using jumpbuffer_t = std::jmp_buf;
            
            public:
                struct ErrorManager {
                    mutable jpeg_error_mgr_t mgr;
                    mutable jumpbuffer_t jumpbuffer;
                    mutable char message[JMSG_LENGTH_MAX];
                    
                    ErrorManager() {
                        jpeg_std_error(&mgr);
                        mgr.error_exit = [](j_common_ptr cinfo) {
                            using ErrorManager = JPEGCompressionBase::ErrorManager;
                            ErrorManager* error_state_ptr = reinterpret_cast<ErrorManager*>(cinfo->err);
                            (*cinfo->err->format_message)(cinfo, error_state_ptr->message);
                            std::longjmp(error_state_ptr->jumpbuffer, 1);
                        };
                        message[0] = 0;
                    }
                    
                } error_state;
            
            public:
                virtual ~JPEGCompressionBase() {}
                
                bool has_error() const {
                    return setjmp(error_state.jumpbuffer);
                }
                
                bool has_warnings() const {
                    return error_state.mgr.num_warnings > 0;
                }
                
                std::string error_message() const {
                    return std::string(error_state.message);
                }
            
        };
        
        /// JPEG decompressor API class, wrapping the `jpeg_decompress_struct`
        /// from jpeglib, alongside a JPEGSourceAdapter instance (q.v. implementation,
        /// above) and furnishing accessor shortcut methods.
        struct JPEGDecompressor : public JPEGCompressionBase final {
            
            enum struct scaling : uint8_t {
                NONE            = 1,
                ONE_HALF        = 2,
                ONE_QUARTER     = 4,
                ONE_EIGHTH      = 8
            };
            
            JPEGDecompressor(byte_source* source)
                :adaptor(source)
                {
                    jpeg_create_decompress(&decompress_state);
                    decompress_state.err = &error_state.mgr;
                    decompress_state.src = &adaptor.mgr;
                }
            
            JPEGDecompressor(void) = delete;
            JPEGDecompressor(JPEGDecompressor const&) = delete;
            JPEGDecompressor(JPEGDecompressor&&) = delete;
            
            virtual ~JPEGDecompressor() {
                if (decompress_started) { jpeg_finish_decompress(&decompress_state); }
                jpeg_destroy_decompress(&decompress_state);
            }
            
            void set_scaling(scaling scale_denominator = scaling::NONE) {
                decompress_state.scale_num = 1;
                decompress_state.scale_denom = int(scale_denominator);
            }
            
            void set_quantize_colors(bool quantize_colors = true) {
                decompress_state.quantize_colors = quantize_colors ? BOOLEAN_TRUE()
                                                                   : BOOLEAN_FALSE();
            }
            
            void set_desired_number_of_colors(int desired_number_of_colors = kDefaultQuantization) {
                decompress_state.desired_number_of_colors = desired_number_of_colors;
            }
            
            void set_quantize_two_pass(bool two_pass_quantize = true) {
                decompress_state.two_pass_quantize = two_pass_quantize ? BOOLEAN_TRUE()
                                                                       : BOOLEAN_FALSE();
            }
            
            void set_dither_mode(J_DITHER_MODE dither_mode = JDITHER_FS) {
                /// other options are JDITHER_NONE and JDITHER_ORDERED --
                /// default (JDITHER_FS) is Floyd-Steinberg dithering:
                decompress_state.dither_mode = dither_mode;
            }
            
            void set_quantization(bool quantization = true, int colorcount = kDefaultQuantization) {
                if (quantization) {
                    set_quantize_colors(true);
                    set_desired_number_of_colors(colorcount);
                    set_quantize_two_pass(true);
                    set_dither_mode(JDITHER_FS);
                    decompress_state.out_color_space = JCS_RGB;
                } else {
                    set_quantize_colors(false);
                }
            }
            
            void set_fancy_upsampling(bool do_fancy_upsampling = true) {
                decompress_state.do_fancy_upsampling = do_fancy_upsampling ? BOOLEAN_TRUE()
                                                                           : BOOLEAN_FALSE();
            }
            
            void set_block_smoothing(bool do_block_smoothing = true) {
                decompress_state.do_block_smoothing = do_block_smoothing ? BOOLEAN_TRUE()
                                                                         : BOOLEAN_FALSE();
            }
            
            bool start(bool read_icc = false) {
                /// see “iccjpeg” functions for documentation of the non-standard
                /// JPEG functions setup_read_icc_profile(…) and read_icc_profile(…):
                if (read_icc) { setup_read_icc_profile(&decompress_state); }
                int const header_status = jpeg_read_header(&decompress_state, BOOLEAN_TRUE());
                if (read_icc && header_status == JPEG_HEADER_OK) {
                    JOCTET* icc_data;
                    unsigned int icc_length;
                    boolean const icc_profile_status = read_icc_profile(&decompress_state,
                                                                        &icc_data,
                                                                        &icc_length);
                    if (icc_profile_status == BOOLEAN_TRUE()) {
                        icc_profile = std::make_unique<JOCTET[]>(icc_length);
                        std::memcpy(icc_profile.get(), icc_data, icc_length);
                        std::free(icc_data);
                    }
                }
                decompress_started = jpeg_start_decompress(&decompress_state);
                return decompress_started && header_status == JPEG_HEADER_OK;
            }
            
            int height() const                  { return decompress_state.output_height;                }
            int width() const                   { return decompress_state.output_width;                 }
            J_COLOR_SPACE _color_space() const  { return color_space_for_components(
                                                         decompress_state.output_components);           }
            J_COLOR_SPACE color_space() const   { return decompress_state.out_color_space;              }
            
            int bit_depth() const               { return decompress_state.data_precision;               }
            int components() const              { return decompress_state.output_components;            }
            int quantized_colors() const        { return decompress_state.actual_number_of_colors;      }
            int scanline() const                { return decompress_state.output_scanline;              }
            
            bool is_multiscan() const           { return jpeg_has_multiple_scans(
                                              const_cast<jpeg_decompress_struct*>(&decompress_state));  }
            bool is_progressive() const         { return decompress_state.progressive_mode;             }
            bool is_quantized() const           { return decompress_state.quantize_colors;              }
            bool is_jfif() const                { return decompress_state.saw_JFIF_marker;              }
            bool is_adobe() const               { return decompress_state.saw_Adobe_marker;             }
            
            byte_ptr colormap() const {
                if (!quantized_colors() || !is_quantized()) {
                    return byte_ptr{ nullptr };
                }
                int colors = quantized_colors();
                int components = decompress_state.out_color_components;
                int mapsize = components * colors;
                byte_ptr output{ std::make_unique<JOCTET[]>(mapsize) };
                for (int idx = 0; idx < components; ++idx) {
                    std::memcpy(output.get() + ptrdiff_t(colors * idx),
                                decompress_state.colormap[idx],
                                colors);
                }
                return output;
            }
            
            JSAMPARRAY allocate_samples(int components = 0,
                                        int width = 0,
                                        int height = 0) {
                /// blow up for invalid heights:
                if (height < 0) {
                    imread_raise(JPEGIOError,
                       "\tim::(anon)::JPEGDecompressor::allocate_samples() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
                    FF("\tim::(anon)::JPEGDecompressor::allocate_samples() got:    `height` = (int){ %i }", height),
                       "\tim::(anon)::JPEGDecompressor::allocate_samples() needs:  `height` > (int){ 0 }");
                }
                /// default to values read from JPEG header:
                if (!width)      {      width = decompress_state.output_width;      }
                if (!height)     {     height = decompress_state.output_height;     }
                if (!components) { components = decompress_state.output_components; }
                /// allocate using internal function pointer:
                return (*decompress_state.mem->alloc_sarray)(reinterpret_cast<j_common_ptr>(&decompress_state),
                                                                              JPOOL_IMAGE,
                                                                              width * components,
                                                                              height);
            }
            
            JSAMPARRAY allocate_samples(J_COLOR_SPACE color_space,
                                        int width = 0,
                                        int height = 0) {
                /// convert the color space constant,
                /// and delegate to the all-ints version above:
                return allocate_samples(components_for_color_space(color_space),
                                        width,
                                        height);
            }
            
            void read_scanlines(JSAMPLE** rows, int idx = 1) {
                jpeg_read_scanlines(&decompress_state, rows, idx);
            }
            
            void read_scanlines(JSAMPLE* rows, int idx = 1) {
                jpeg_read_scanlines(&decompress_state, std::addressof(rows), idx);
            }
            
            public:
                JPEGSourceAdaptor adaptor;
                jpeg_decompress_struct decompress_state;
                bool decompress_started{ false };
                byte_ptr icc_profile{ nullptr };
        };
        
        /// JPEG compressor API class, wrapping the `jpeg_compress_struct`
        /// from jpeglib, alongside a JPEGDestinationAdapter instance (q.v. implementation,
        /// above) and furnishing accessor shortcut methods.
        struct JPEGCompressor : public JPEGCompressionBase final {
            
            JPEGCompressor(byte_sink* sink)
                :adaptor(sink)
                {
                    jpeg_create_compress(&compress_state);
                    compress_state.err = &error_state.mgr;
                    compress_state.dest = &adaptor.mgr;
                }
            
            JPEGCompressor(void) = delete;
            JPEGCompressor(JPEGCompressor const&) = delete;
            JPEGCompressor(JPEGCompressor&&) = delete;
            
            virtual ~JPEGCompressor() {
                if (compress_started) { jpeg_finish_compress(&compress_state); }
                jpeg_destroy_compress(&compress_state);
            }
            
            void start() {
                jpeg_start_compress(&compress_state, BOOLEAN_TRUE());
                compress_started = true;
            }
            
            void set_defaults() {
                jpeg_set_defaults(&compress_state);
            }
            
            void set_quality(std::size_t quality = kDefaultQuality) {
                if (quality > 100) { quality = 100; }
                jpeg_set_quality(&compress_state, quality, BOOLEAN_FALSE());
            }
            
            void set_optimal_coding(bool optimal_coding = true) {
                compress_state.optimize_coding = optimal_coding ? BOOLEAN_TRUE()
                                                                : BOOLEAN_FALSE();
            }
            
            int height() const                  { return compress_state.image_height; }
            int width() const                   { return compress_state.image_width; }
            J_COLOR_SPACE color_space() const   { return color_space_for_components(
                                                         compress_state.input_components); }
            int components() const              { return compress_state.input_components; }
            int next_scanline() const           { return compress_state.next_scanline; }
            
            void set_height(int height)         { compress_state.image_height = height; }
            void set_width(int width)           { compress_state.image_width = width; }
            void set_color_space(J_COLOR_SPACE color_space) {
                                                  compress_state.input_components = 
                                                       components_for_color_space(color_space);
                                                  compress_state.in_color_space = color_space; }
            void set_components(int components) { compress_state.input_components = components;
                                                  compress_state.in_color_space =
                                                         color_space_for_components(components); }
            
            void set_icc_profile(JOCTET const* icc_data, unsigned int icc_size) {
                /// Call this BEFORE jpeg_write_scanlines()!
                write_icc_profile(&compress_state,
                                   icc_data,
                                   icc_size);
            }
            
            void set_icc_profile(const void* icc_data, std::size_t icc_size) {
                /// Call this BEFORE jpeg_write_scanlines()!
                write_icc_profile(&compress_state,
                                   static_cast<JOCTET const*>(icc_data),
                                   static_cast<unsigned int>(icc_size));
            }
            
            JSAMPARRAY allocate_samples(int components = 0,
                                        int width = 0,
                                        int height = 0) {
                /// blow up for invalid heights:
                if (height < 0) {
                    imread_raise(JPEGIOError,
                       "\tim::(anon)::JPEGCompressor::allocate_samples() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"",
                    FF("\tim::(anon)::JPEGCompressor::allocate_samples() got:    `height` = (int){ %i }", height),
                       "\tim::(anon)::JPEGCompressor::allocate_samples() needs:  `height` > (int){ 0 }");
                }
                /// default to values attached to ‘info’ struct:
                if (!width)      {      width = compress_state.image_width;      }
                if (!height)     {     height = compress_state.image_height;     }
                if (!components) { components = compress_state.input_components; }
                /// allocate using internal function pointer:
                return (*compress_state.mem->alloc_sarray)(reinterpret_cast<j_common_ptr>(&compress_state),
                                                                            JPOOL_IMAGE,
                                                                            width * components,
                                                                            height);
            }
            
            JSAMPARRAY allocate_samples(J_COLOR_SPACE color_space,
                                        int width = 0,
                                        int height = 0) {
                /// convert the color space constant,
                /// and delegate to the all-ints version above:
                return allocate_samples(components_for_color_space(color_space),
                                        width,
                                        height);
            }
            
            void write_scanlines(JSAMPLE** rows, int idx = 1) {
                jpeg_write_scanlines(&compress_state, rows, idx);
            }
            
            void write_scanlines(JSAMPLE* rows, int idx = 1) {
                jpeg_write_scanlines(&compress_state, std::addressof(rows), idx);
            }
            
            public:
                JPEGDestinationAdaptor adaptor;
                jpeg_compress_struct compress_state;
                bool compress_started{ false };
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
        bool read_metadata_inline = opts.cast<bool>("jpg:read_metadata", true);
        bool read_icc_profile = opts.cast<bool>("jpg:read_icc_profile", false);
        
        /// first, read the JPEG header, the image metadata,
        /// and (optionally) the embedded ICC profile data:
        decompressor.start(read_icc_profile);
        
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
        const int b = decompressor.bit_depth();
        
        /// sanity-check JPEG source’s bit depth:
        if (b != 8) {
            imread_raise(JPEGIOError,
                FF("JPEG bit depth must be 8 (got bit_depth = %i)", b));
        }
        
        /// create the output image:
        std::unique_ptr<Image> output(factory->create(b, h, w, c));
        
        /// allocate a single-row sample array:
        JSAMPARRAY samples = decompressor.allocate_samples();
        
        /// Hardcoding JSAMPLE (== uint8_t) as the type for now:
        int color_stride = output->stride_or(2, 0);
        JSAMPLE* __restrict__ ptr = output->rowp_as<JSAMPLE>(0);
        
        /// read image data as scanlines, in a loop:
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
        
        /// add metadata:
        if (read_metadata_inline) {
            output->metadata(read_metadata(src, opts));
        }
        
        return output;
    }
    
    Metadata JPEGFormat::read_metadata(byte_source* src,
                                       Options const& opts) {
        using easyexif::EXIFInfo;
        using im::byte_iterator;
        
        Metadata meta;
        
        byte_iterator result = std::search(src->begin(),   src->end(),
                                           marker.begin(), marker.end());
        bool has_exif = result != src->end();
        if (!has_exif) { return meta; }
        
        bytevec_t rawbytes;
        uint16_t size = parse_size(result);
        rawbytes.reserve(size);
        
        std::advance(result, 4);
        std::copy(result, result + size,
                  std::back_inserter(rawbytes));
        
        EXIFInfo exif;
        if (exif.parseFromEXIFSegment(rawbytes.data(), rawbytes.size()) != PARSE_EXIF_SUCCESS) {
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
        
        /// sanity-check input bit depth:
        if (input.nbits() != 8) {
            imread_raise(CannotWriteError,
                FF("Image must be 8 bits for JPEG saving (got %i)",
                    input.nbits()));
        }
        
        JPEGCompressor compressor(output);
        
        const int w = input.dim(0);
        const int h = input.dim(1);
        const int c = std::min(3, input.dim_or(2));
        
        /// assign image values:
        compressor.set_width(w);
        compressor.set_height(h);
        compressor.set_components(c);
        
        /// optimize Huffman coding tables:
        compressor.set_optimal_coding();
        
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
        av::strided_array_view<JSAMPLE, 1> subview;
        
        /// write image data as scanlines, in a pixel loop:
        int scan;
        while ((scan = compressor.next_scanline()) < h) {
            JSAMPLE* __restrict__ dstPtr = samples[0];
            for (int x = 0; x < w; ++x) {
                subview = view[x][scan];
                for (int cc = 0; cc < c; ++cc) {
                    *dstPtr++ = subview[cc];
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
