/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#define NO_IMPORT_ARRAY

#include <iod/json.hh>
#include <libimread/errors.hh>
#include <libimread/metadata.hh>
#include <libimread/ext/base64.hh>
#include <libimread/IO/tiff.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>

extern "C" {
   #include <tiffio.h>
}

namespace im {
    
    DECLARE_FORMAT_OPTIONS(STKFormat);
    DECLARE_FORMAT_OPTIONS(TIFFFormat);
    
    /// TIFF read function pointer type:
    using tiff_read_f = std::add_pointer_t<tsize_t(thandle_t, void*, tsize_t)>;
    
    /// Buffer size constant:
    const std::size_t kBufferSize = TIFFFormat::options.buffer_size;
    
    namespace {
        
        tsize_t tiff_read(thandle_t handle, void* data, tsize_t n) {
            byte_source* source = static_cast<byte_source*>(handle);
            return source->read(static_cast<byte*>(data), n);
        }
        
        tsize_t tiff_read_from_writer(thandle_t handle, void* data, tsize_t n) {
            byte_sink* sink = static_cast<byte_sink*>(handle);
            byte_source* source = dynamic_cast<byte_source*>(sink);
            if (!source) {
                imread_raise(ProgrammingError,
                    "Could not dynamic_cast<> to byte_source");
            }
            return source->read(static_cast<byte*>(data), n);
        }
        
        tsize_t tiff_write(thandle_t handle, void* data, tsize_t n) {
            byte_sink* sink = static_cast<byte_sink*>(handle);
            return sink->write(static_cast<byte*>(data), n);
        }
        
        tsize_t tiff_no_read(thandle_t, void*, tsize_t) {
            return 0;
        }
        
        tsize_t tiff_no_write(thandle_t, void*, tsize_t) {
            imread_raise(ProgrammingError,
                "tiff_no_write() called during read");
        }
        
        template <typename Seekable>
        toff_t tiff_seek(thandle_t handle, toff_t off, int whence) {
            using pointer_t = std::add_pointer_t<Seekable>;
            pointer_t seekable = static_cast<pointer_t>(handle);
            switch (whence) {
                case SEEK_SET: return seekable->seek_absolute(off);
                case SEEK_CUR: return seekable->seek_relative(off);
                case SEEK_END: return seekable->seek_end(off);
            }
            return -1;
        }
        
        template <typename Seekable>
        toff_t tiff_size(thandle_t handle) {
            using pointer_t = std::add_pointer_t<Seekable>;
            pointer_t seekable = static_cast<pointer_t>(handle);
            const std::size_t curpos = seekable->seek_relative(0);
            const std::size_t size = seekable->seek_end(0);
            seekable->seek_absolute(curpos);
            return toff_t(size);
        }
        
        template <>
        toff_t tiff_size<byte_source>(thandle_t handle) {
            using pointer_t = std::add_pointer_t<byte_source>;
            pointer_t source = static_cast<pointer_t>(handle);
            return toff_t(source->size());
        }
        
        int tiff_close(thandle_t handle) { return 0; }
        
        struct TIFFWrapperBase {
            TIFF* tif;
            
            TIFFWrapperBase(TIFF* t)
                :tif(t)
                {}
            
            ~TIFFWrapperBase() {
                // TIFFClose(tif);
            }
            
        };
        
        class TIFFWarningsAndErrors {
            
            public:
                /// Newer versions of TIFF seem to call this TIFFWarningHandler,
                /// but older versions do not have this type
                using tiff_handler_f = std::add_pointer_t<void(char const* module,
                                                               char const* fmt,
                                                               va_list ap)>;
            
            public:
                TIFFWarningsAndErrors()
                      
                      :warning_handle(TIFFSetWarningHandler([](const char* module,
                                                               const char* fmt,
                                                               va_list ap) { std::fprintf(stderr, "[TIFF/WARNING] %s: ", module);
                                                                             std::vfprintf(stderr, fmt, ap);
                                                                             std::fprintf(stderr, "\n"); }))
                          
                          ,error_handle(TIFFSetErrorHandler([](const char* module,
                                                               const char* fmt,
                                                               va_list ap) { char buffer[kBufferSize];
                                                                             std::vsnprintf(buffer, sizeof(char) * kBufferSize, fmt, ap);
                                                                             imread_raise(TIFFIOError, "FATAL in TIFF I/O",
                                                                                 FF("[TIFF/ERROR***] %s: ", module),
                                                                                 FF("%s\n", std::string(buffer).c_str())); }))
                    {}
            
            public:
                ~TIFFWarningsAndErrors() { TIFFSetWarningHandler(warning_handle);
                                           TIFFSetErrorHandler(error_handle); }
            
            protected:
                tiff_handler_f warning_handle;
                tiff_handler_f error_handle;
            
        };
        
        template <typename ReadType>
        ReadType tiff_get(TIFFWrapperBase const& t, const int tag) {
            ReadType val;
            if (!TIFFGetField(t.tif, tag, &val)) {
                imread_raise(TIFFIOError,
                    "Could not get tag:",
                 FF("\t>>> %s", tag));
            }
            return val;
        }
        
        template <typename ReadType>
        ReadType tiff_get(TIFFWrapperBase const& t, const int tag,
                                               const ReadType default_value) {
            ReadType val;
            if (!TIFFGetField(t.tif, tag, &val)) {
                return default_value;
            }
            return val;
        }
        
        template <>
        std::string tiff_get<std::string>(TIFFWrapperBase const& t, const int tag,
                                          const std::string default_value) {
            char* val;
            if (!TIFFGetField(t.tif, tag, &val)) {
                return default_value;
            }
            return val;
        }
        
        TIFF* read_client(byte_source* src) {
            return TIFFClientOpen(
                            "internal",
                            "r",
                            src,
                            tiff_read,
                            tiff_no_write,
                            tiff_seek<byte_source>,
                            tiff_close,
                            tiff_size<byte_source>,
                            nullptr,
                            nullptr);
        }
        
        const int UIC1Tag = 33628;
        const int UIC2Tag = 33629;
        const int UIC3Tag = 33630;
        const int UIC4Tag = 33631;
        
        std::array<TIFFFieldInfo, 5> stkTags{{
            { UIC1Tag, -1,-1,   TIFF_LONG,        FIELD_CUSTOM, true, true,   const_cast<char*>("UIC1Tag") },
            { UIC1Tag, -1,-1,   TIFF_RATIONAL,    FIELD_CUSTOM, true, true,   const_cast<char*>("UIC1Tag") },
          //{ UIC2Tag, -1, -1,  TIFF_RATIONAL,    FIELD_CUSTOM, true, true,   const_cast<char*>("UIC2Tag") },
            { UIC2Tag, -1, -1,  TIFF_LONG,        FIELD_CUSTOM, true, true,   const_cast<char*>("UIC2Tag") },
            { UIC3Tag, -1,-1,   TIFF_RATIONAL,    FIELD_CUSTOM, true, true,   const_cast<char*>("UIC3Tag") },
            { UIC4Tag, -1,-1,   TIFF_LONG,        FIELD_CUSTOM, true, true,   const_cast<char*>("UIC4Tag") },
        }};
        
        template <typename ByteSource,
                  typename = std::enable_if_t<
                             std::is_base_of<ByteSource,
                                             im::byte_source>::value>>
        class SourceShifter : public ByteSource {
            
            public:
                using type = ByteSource;
            
            public:
                explicit SourceShifter(ByteSource* s)
                    :source(s)
                    ,shift_by{ 0 }
                    {}
                
                explicit SourceShifter(ByteSource* s, int nshift)
                    :source(s)
                    ,shift_by{ nshift }
                    { source->seek_relative(nshift); }
                
            public:
                virtual std::size_t seek_absolute(std::size_t pos)          { return source->seek_absolute(pos + shift_by) - shift_by; }
                virtual std::size_t seek_relative(int n)                    { return source->seek_relative(n) - shift_by; }
                virtual std::size_t seek_end(int n)                         { return source->seek_end(n + shift_by) - shift_by; }
                virtual std::size_t read(byte* buf, std::size_t n) const    { return source->read(buf, n); }
                
            public:
                /// delegate these fully, to avoid
                /// invoking the naive implementations:
                virtual bytevec_t full_data() const                         { return source->full_data(); }
                virtual std::size_t size() const                            { return source->size(); }
                virtual void* readmap(std::size_t pageoffset = 0) const     { return source->readmap(pageoffset); }
                
            public:
                /// non-override setter assigns the shift:
                void shift(int nshift) {
                    source->seek_relative(nshift - shift_by);
                    shift_by = nshift;
                }
                
            protected:
                ByteSource* source;
                int shift_by;
                
            private:
                SourceShifter(SourceShifter const&);
                SourceShifter(SourceShifter&&);
        };
        
        using ShiftSource = SourceShifter<im::byte_source>;
        
        class STKTagExtender {
            
            public:
                STKTagExtender()
                    :proc(TIFFSetTagExtender([](TIFF* tif) -> void {
                          TIFFMergeFieldInfo(tif,
                                             stkTags.data(),
                                             stkTags.size()); }))
                    {}
            
            public:
                ~STKTagExtender() { TIFFSetTagExtender(proc); }
            
            protected:
                TIFFExtendProc proc;
        };
        
    } /// namespace (anon.)
    
    ImageList STKFormat::do_read(byte_source* src, ImageFactory* factory,
                                                   bool is_multi,
                                                   Options const& opts) {
        
        STKTagExtender ext;
        TIFFWarningsAndErrors twe;
        std::unique_ptr<ShiftSource> moved = std::make_unique<ShiftSource>(src);
        TIFFWrapperBase t = read_client(moved.get());
        ImageList images;
        
        const uint32_t h                    = tiff_get<uint32_t>(t, TIFFTAG_IMAGELENGTH);
        const uint32_t w                    = tiff_get<uint32_t>(t, TIFFTAG_IMAGEWIDTH);
        const uint16_t nr_samples           = tiff_get<uint16_t>(t, TIFFTAG_SAMPLESPERPIXEL, options.readopts.samples_per_pixel);
        const uint16_t bits_per_sample      = tiff_get<uint16_t>(t, TIFFTAG_BITSPERSAMPLE,   options.readopts.bits_per_sample);
        
        const int depth = nr_samples > 1 ? nr_samples : 1;
        const int strip_size = TIFFStripSize(t.tif);
        const int n_strips = TIFFNumberOfStrips(t.tif);
        int raw_strip_size = 0;
        int z = 0;
        
        int32_t n_planes;
        void* __restrict__ data; /// UNUSED, WAT
        
        TIFFGetField(t.tif, UIC3Tag, &n_planes, &data);
        
        for (int st = 0; st != n_strips; ++st) {
            raw_strip_size += TIFFRawStripSize(t.tif, st);
        }
        
        /// ORIGINALLY: for (int z = 0; z < n_planes; ++z) {…}
        
        do {
            /// Monkey patch strip offsets -- 
            /// This is very hacky, but it seems to work!
            moved->shift(z * raw_strip_size);
            std::unique_ptr<Image> output(factory->create(bits_per_sample, h, w, depth));
            
            // if (Metadata* metaout = dynamic_cast<Metadata*>(output.get())) {
            //     std::string description = tiff_get<std::string>(t, TIFFTAG_IMAGEDESCRIPTION,
            //                                                        options.metadata);
            //     if (description.size()) {
            //         metaout->set_meta(description);
            //     }
            // }
            
            byte* start = output->rowp_as<byte>(0);
            for (int st = 0; st != n_strips; ++st) {
                const int offset = TIFFReadEncodedStrip(t.tif, st, start, strip_size);
                if (offset == -1) {
                    imread_raise(TIFFIOError,
                        "Error reading encoded STK strip");
                }
                start += offset;
            }
            
            /// Add image to image list for return:
            images.push_back(std::move(output));
            
            /// Increment image index (z):
            ++z;
            
        } while (is_multi && z < n_planes);
        
        TIFFClose(t.tif);
        
        return images;
    }
    
    bool TIFFFormat::match_format(byte_source* src) {
        /// Match either second or third four-byte magic signature:
        return match_magic(src, base64::decode(options.signatures[1].bytes).get(), 
                                               options.signatures[1].length) ||
               match_magic(src, base64::decode(options.signatures[2].bytes).get(), 
                                               options.signatures[2].length);
    }
    
    ImageList TIFFFormat::do_read(byte_source* src, ImageFactory* factory,
                                                    bool is_multi,
                                                    Options const& opts) {
        TIFFWarningsAndErrors twe;
        TIFFWrapperBase t = read_client(src);
        ImageList images;
        
        const uint32_t h                = tiff_get<uint32_t>(t, TIFFTAG_IMAGELENGTH);
        const uint32_t w                = tiff_get<uint32_t>(t, TIFFTAG_IMAGEWIDTH);
        const uint16_t nr_samples       = tiff_get<uint16_t>(t, TIFFTAG_SAMPLESPERPIXEL);
        const uint16_t bits_per_sample  = tiff_get<uint16_t>(t, TIFFTAG_BITSPERSAMPLE);
        const int depth = nr_samples > 1 ? nr_samples : 1;
        
        do {
            std::unique_ptr<Image> output = factory->create(bits_per_sample, h, w, depth);
            
            // if (Metadata* meta = dynamic_cast<Metadata*>(output.get())) {
            //     std::string description = tiff_get<std::string>(t, TIFFTAG_IMAGEDESCRIPTION,
            //                                                        options.metadata);
            //     if (description.size()) {
            //         meta->set_meta(description);
            //     }
            // }
            
            if (bits_per_sample == 8) {
                /// Hardcoding uint8_t as the type for now
                int c_stride = (depth == 1) ? 0 : output->stride(2);
                
                for (int r = 0; r != h; ++r) {
                    /// NB. This is highly specious
                    byte* ptr = output->rowp_as<byte>(r);
                    byte* dstPtr = output->rowp_as<byte>(r);
                    if (TIFFReadScanline(t.tif, ptr, r) == -1) {
                        imread_raise(TIFFIOError, "Error in pixel read loop", FF("scanline = %i", r));
                    }
                    /// this pixel loop de-interleaves the destination buffer in-place
                    for (int x = 0; x < w; ++x) {
                        for (int c = 0; c < depth; ++c) {
                            dstPtr[c*c_stride] = *ptr++;
                        }
                        dstPtr++;
                    }
                }
                
            } else if (bits_per_sample == 16) {
                /// Hardcoding uint16_t as the type for now
                int c_stride = (depth == 1) ? 0 : output->stride(2);
                uint16_t* ptr16 = (uint16_t*)std::calloc(sizeof(uint16_t), TIFFScanlineSize(t.tif)*3);
                
                WTF("About to enter 16-bit pixel loop...");
                
                for (int r = 0; r != h; ++r) {
                    byte* dstPtr = output->rowp_as<byte>(r);
                    if (TIFFReadScanline(t.tif, ptr16, r) == -1) {
                        std::free(ptr16);
                        imread_raise(TIFFIOError, "Error in pixel read loop", FF("scanline = %i", r));
                    }
                    /// this pixel loop de-interleaves the destination buffer in-place
                    for (int x = 0; x < w; ++x) {
                        for (int c = 0; c < depth; ++c) {
                            uint16_t hi = (*ptr16++) << 8;
                            uint16_t lo = hi | (*ptr16++);
                            dstPtr[c*c_stride] = lo;
                        }
                        dstPtr++;
                    }
                }
                std::free(ptr16);
            }
            
            images.push_back(std::move(output));
        
        } while (is_multi && TIFFReadDirectory(t.tif));
        
        TIFFClose(t.tif);
        return images;
    }
    
    void TIFFFormat::write(Image& input, byte_sink* output,
                                         Options const& opts) {
        TIFFWarningsAndErrors twe;
        TIFFWrapperBase t = TIFFClientOpen(
                        "internal",
                        "w",
                        output,
                        tiff_no_read,
                        tiff_write,
                        tiff_seek<byte_sink>,
                        tiff_close,
                        tiff_size<byte_sink>,
                        nullptr,
                        nullptr);
        
        bytevec_t bufdata;
        byte* __restrict__ bufp = 0;
        bool copy_data = opts.cast<bool>(
                         "tiff:copy-data",
                         options.writeopts.copy_data);
        const uint32_t w = input.dim(0);
        const uint32_t h = input.dim(1);
        const uint32_t ch = input.dim(2);
        const uint32_t siz = input.size();
        const uint32_t nbytes = input.nbytes();
        const uint16_t photometric = ((input.ndims() == 3 && ch) ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
        
        TIFFSetField(t.tif, TIFFTAG_IMAGELENGTH,        static_cast<uint32_t>(h));
        TIFFSetField(t.tif, TIFFTAG_IMAGEWIDTH,         static_cast<uint32_t>(w));
        
        TIFFSetField(t.tif, TIFFTAG_BITSPERSAMPLE,      static_cast<uint16_t>(input.nbits()));
        TIFFSetField(t.tif, TIFFTAG_SAMPLESPERPIXEL,    static_cast<uint16_t>(input.dim_or(2, 1)));
        
        TIFFSetField(t.tif, TIFFTAG_PHOTOMETRIC,        static_cast<uint16_t>(photometric));
        TIFFSetField(t.tif, TIFFTAG_PLANARCONFIG,       PLANARCONFIG_CONTIG);
        
        if (opts.cast<bool>(
            "tiff:compress",
            options.writeopts.compress)) {
            TIFFSetField(t.tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
            // For 8 bit images, prediction defaults to false; for 16 bit images,
            // it defaults to true. This is because compression of raw 16 bit
            // images is often counter-productive without this flag. See the
            // discusssion at http://www.asmail.be/msg0055176395.html
            const bool prediction_default = input.nbits() != 8;
            if (opts.cast<bool>(
                "tiff:horizontal-predictor",
                options.writeopts.horizontal_predictor)) {
                TIFFSetField(t.tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
                if (!copy_data) { copy_data = true; }
            }
        }
        
        /// NB. Get this from Image object Metadata ancestor -- or else
        /// why the fuck are we even using that shit?
        if (opts.cast<bool>(
            "tiff:metadata",
            options.writeopts.metadata)) {
            std::string meta = opts.cast<std::string>(
                               "metadata",
                               options.metadata);
            std::string ssig = opts.cast<std::string>(
                               "tiff:software-signature",
                               options.writeopts.software_signature);
            
            TIFFSetField(t.tif, TIFFTAG_IMAGEDESCRIPTION, meta.c_str());
            TIFFSetField(t.tif, TIFFTAG_SOFTWARE, ssig.c_str());
            
            TIFFSetField(t.tif, TIFFTAG_XRESOLUTION,
                opts.cast<int>(
                "tiff:x-resolution",
                options.writeopts.x_resolution));
            TIFFSetField(t.tif, TIFFTAG_YRESOLUTION,
                opts.cast<int>(
                "tiff:y-resolution",
                options.writeopts.y_resolution));
            TIFFSetField(t.tif, TIFFTAG_RESOLUTIONUNIT,
                opts.cast<int>(
                "tiff:resolution-unit", RESUNIT_INCH));
            TIFFSetField(t.tif, TIFFTAG_ORIENTATION,
                opts.cast<int>(
                "tiff:orientation", ORIENTATION_TOPLEFT));
        }
        
        if (copy_data) {
            bufdata.resize(siz * nbytes);
            bufp = &bufdata[0];
            
            if (ch == 1) {
                std::memcpy(bufp, input.rowp(0), siz * nbytes);
            } else if (ch == 3) {
                byte* __restrict__ srcp;
                byte* __restrict__ pixel;
                byte* __restrict__ R;
                byte* __restrict__ G;
                byte* __restrict__ B;
                int r = 0,
                    x = 0;
                
                /// pre-interlace buffer:
                int cs1 = input.stride(2);
                int cs2 = cs1 + cs1;
                for (r = 0; r < h; ++r) {
                    pixel = &bufp[r * h * nbytes];
                    R = pixel++;
                    G = pixel++;
                    B = pixel;
                    srcp = input.rowp_as<byte>(r);
                    for (x = 0; x < w; ++x) {
                        *R = *srcp++;
                        R += ch;
                    }
                    srcp = input.rowp_as<byte>(r) + cs1;
                    for (x = 0; x < w; ++x) {
                        *G = *srcp++;
                        G += ch;
                    }
                    srcp = input.rowp_as<byte>(r) + cs2;
                    for (x = 0; x < w; ++x) {
                        *B = *srcp++;
                        B += ch;
                    }
                }
                
            }
            
        }
        
        for (int rr = 0; rr != h; ++rr) {
            void* __restrict__ rowp = copy_data ? bufp + (rr * h * nbytes) : input.rowp(rr);
            if (TIFFWriteScanline(t.tif, rowp, rr) == -1) {
                imread_raise(TIFFIOError, "Error writing scanline");
            }
        }
        
        TIFFFlush(t.tif);
        TIFFClose(t.tif);
    }
    
    void TIFFFormat::write_multi(ImageList& input, byte_sink* output,
                                                   Options const& opts) {
        do_write(input, output, true, opts);
    }
    
    void TIFFFormat::do_write(ImageList& input, byte_sink* output,
                                                bool is_multi,
                                                Options const& opts) {
        TIFFWarningsAndErrors twe;
        tiff_read_f read_function = dynamic_cast<byte_source*>(output) ?
                                    tiff_read_from_writer :
                                    tiff_no_read;
        
        TIFFWrapperBase t = TIFFClientOpen(
                        "internal",
                        "w",
                        output,
                        read_function, /// read_function
                        tiff_write,
                        tiff_seek<byte_sink>,
                        tiff_close,
                        tiff_size<byte_sink>,
                        nullptr,
                        nullptr);
        
        bytevec_t bufdata;
        const unsigned n_pages = input.size();
        
        if (is_multi) {
            TIFFCreateDirectory(t.tif);
        }
        
        for (int i = 0; i != n_pages; ++i) {
            Image* im = input.at(i);
            byte* __restrict__ bufp = 0;
            bool copy_data = opts.cast<bool>(
                             "tiff:copy-data",
                             options.writeopts.copy_data);
            const uint32_t w = im->dim(0);
            const uint32_t h = im->dim(1);
            const uint32_t ch = im->dim(2);
            const uint32_t siz = im->size();
            const uint32_t nbytes = im->nbytes();
            const uint16_t photometric = ((im->ndims() == 3 && ch) ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
            
            TIFFSetField(t.tif, TIFFTAG_IMAGELENGTH,        static_cast<uint32_t>(h));
            TIFFSetField(t.tif, TIFFTAG_IMAGEWIDTH,         static_cast<uint32_t>(w));
            
            TIFFSetField(t.tif, TIFFTAG_BITSPERSAMPLE,      static_cast<uint16_t>(im->nbits()));
            TIFFSetField(t.tif, TIFFTAG_SAMPLESPERPIXEL,    static_cast<uint16_t>(im->dim_or(2, 1)));
            
            TIFFSetField(t.tif, TIFFTAG_PHOTOMETRIC,        static_cast<uint16_t>(photometric));
            TIFFSetField(t.tif, TIFFTAG_PLANARCONFIG,       PLANARCONFIG_CONTIG);
            
            if (opts.cast<bool>(
                "tiff:compress",
                options.writeopts.compress)) {
                TIFFSetField(t.tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
                // For 8 bit images, prediction defaults to false; for 16 bit images,
                // it defaults to true. This is because compression of raw 16 bit
                // images is often counter-productive without this flag. See the
                // discusssion at http://www.asmail.be/msg0055176395.html
                const bool prediction_default = im->nbits() != 8;
                if (opts.cast<bool>(
                    "tiff:horizontal-predictor",
                    options.writeopts.horizontal_predictor)) {
                    TIFFSetField(t.tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
                    if (!copy_data) { copy_data = true; }
                }
            }
            
            /// NB. Get this from Image object Metadata ancestor -- or else
            /// why the fuck are we even using that shit?
            if (opts.cast<bool>(
                "tiff:metadata",
                options.writeopts.metadata)) {
                std::string meta = opts.cast<std::string>(
                                   "metadata",
                                   options.metadata);
                std::string ssig = opts.cast<std::string>(
                                   "tiff:software-signature",
                                   options.writeopts.software_signature);
                    
                TIFFSetField(t.tif, TIFFTAG_IMAGEDESCRIPTION, meta.c_str());
                TIFFSetField(t.tif, TIFFTAG_SOFTWARE, ssig.c_str());
                
                TIFFSetField(t.tif, TIFFTAG_XRESOLUTION,
                    opts.cast<int>(
                    "tiff:x-resolution",
                    options.writeopts.x_resolution));
                TIFFSetField(t.tif, TIFFTAG_YRESOLUTION,
                    opts.cast<int>(
                    "tiff:y-resolution",
                    options.writeopts.y_resolution));
                TIFFSetField(t.tif, TIFFTAG_RESOLUTIONUNIT,
                    opts.cast<int>(
                    "tiff:resolution-unit", RESUNIT_INCH));
                TIFFSetField(t.tif, TIFFTAG_ORIENTATION,
                    opts.cast<int>(
                    "tiff:orientation", ORIENTATION_TOPLEFT));
            }
            
            if (is_multi) {
                TIFFSetField(t.tif, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
                TIFFSetField(t.tif, TIFFTAG_PAGENUMBER, i, n_pages);
            }
            
            if (copy_data) {
                bufdata.resize(siz * nbytes);
                bufp = &bufdata[0];
                
                if (ch == 1) {
                    std::memcpy(bufp, im->rowp(0), siz * nbytes);
                } else if (ch == 3) {
                    byte* __restrict__ srcp;
                    byte* __restrict__ pixel;
                    byte* __restrict__ R;
                    byte* __restrict__ G;
                    byte* __restrict__ B;
                    int r = 0,
                        x = 0;
                    
                    /// pre-interlace buffer:
                    int cs1 = im->stride(2);
                    int cs2 = cs1 + cs1;
                    for (r = 0; r < h; ++r) {
                        pixel = &bufp[r * h * nbytes];
                        R = pixel++;
                        G = pixel++;
                        B = pixel;
                        srcp = im->rowp_as<byte>(r);
                        for (x = 0; x < w; ++x) {
                            *R = *srcp++;
                            R += ch;
                        }
                        srcp = im->rowp_as<byte>(r) + cs1;
                        for (x = 0; x < w; ++x) {
                            *G = *srcp++;
                            G += ch;
                        }
                        srcp = im->rowp_as<byte>(r) + cs2;
                        for (x = 0; x < w; ++x) {
                            *B = *srcp++;
                            B += ch;
                        }
                    }
                
                }
            }
            
            for (int rr = 0; rr != h; ++rr) {
                void* __restrict__ rowp = copy_data ? bufp + (rr * h * nbytes) : im->rowp(rr);
                if (TIFFWriteScanline(t.tif, rowp, rr) == -1) {
                    imread_raise(TIFFIOError, "Error writing scanline");
                }
            }
            
            if (is_multi) {
                if (!TIFFWriteDirectory(t.tif)) {
                    imread_raise(TIFFIOError, "TIFFWriteDirectory failed");
                }
            }
        }
        
        TIFFFlush(t.tif);
        TIFFClose(t.tif);
    }
}