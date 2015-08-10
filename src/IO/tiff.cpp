/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#define NO_IMPORT_ARRAY

#include <libimread/IO/tiff.hh>

extern "C" {
   #include <tiffio.h>
}

namespace im {
    
    namespace {
        
        void show_tiff_warning(const char *module, const char *fmt, va_list ap) {
            std::fprintf(stderr, "%s: ", module);
            std::vfprintf(stderr, fmt, ap);
            std::fprintf(stderr, "\n");
        }
        
        tsize_t tiff_read(thandle_t handle, void *data, tsize_t n) {
            byte_source* s = static_cast<byte_source*>(handle);
            return s->read(static_cast<byte*>(data), n);
        }
        
        tsize_t tiff_write(thandle_t handle, void *data, tsize_t n) {
            byte_sink* s = static_cast<byte_sink*>(handle);
            return s->write(static_cast<byte*>(data), n);
        }
        
        tsize_t tiff_no_read(thandle_t, void*, tsize_t) {
            return 0;
        }
        
        tsize_t tiff_no_write(thandle_t, void*, tsize_t) {
            imread_raise(ProgrammingError, "tiff_write called when reading");
        }
        
        template <typename T>
        toff_t tiff_seek(thandle_t handle, toff_t off, int whence) {
            T *s = static_cast<T*>(handle);
            switch (whence) {
                case SEEK_SET: return s->seek_absolute(off);
                case SEEK_CUR: return s->seek_relative(off);
                case SEEK_END: return s->seek_end(off);
            }
            return -1;
        }
        
        int tiff_close(thandle_t handle) { return 0; }
        
        template <typename T>
        toff_t tiff_size(thandle_t handle) {
            T *s = static_cast<T*>(handle);
            const std::size_t curpos = s->seek_relative(0);
            const std::size_t size = s->seek_end(0);
            s->seek_absolute(curpos);
            return size;
        }
        
        void tiff_error(const char *module, const char *fmt, va_list ap) {
            char buffer[4096];
            vsnprintf(buffer, sizeof(buffer), fmt, ap);
            imread_raise(CannotReadError, "TIFF library error:", buffer);
        }
        
        struct tif_holder {
            tif_holder(TIFF *tif)
                :tif(tif)
                { }
            
            ~tif_holder() {
                TIFFClose(tif);
            }
            
            TIFF *tif;
        };
        
        struct tiff_warn_error {
            tiff_warn_error()
                :warning_handler_(TIFFSetWarningHandler(show_tiff_warning))
                ,error_handler_(TIFFSetErrorHandler(tiff_error))
            { }
            ~tiff_warn_error() {
                TIFFSetWarningHandler(warning_handler_);
                TIFFSetErrorHandler(error_handler_);
            }
            
            // Newer versions of TIFF seem to call this TIFFWarningHandler, but older versions do not have this type
            typedef void (*tiff_handler_type)(const char *module, const char *fmt, va_list ap);
            tiff_handler_type warning_handler_;
            tiff_handler_type error_handler_;
        };
        
        template <typename T>
        inline T tiff_get(const tif_holder &t, const int tag) {
            T val;
            if (!TIFFGetField(t.tif, tag, &val)) {
                imread_raise(CannotReadError,
                    "Cannot find necessary tag:",
                 FF("\t>>> %s", tag));
            }
            return val;
        }
    
        template <typename T>
        inline T tiff_get(const tif_holder &t, const int tag, const T def) {
            T val;
            if (!TIFFGetField(t.tif, tag, &val)) { return def; }
            return val;
        }
    
        template <>
        inline std::string tiff_get<std::string>(const tif_holder &t, const int tag, const std::string def) {
            char *val;
            if (!TIFFGetField(t.tif, tag, &val)) { return def; }
            return val;
        }
        
        TIFF *read_client(byte_source *src) {
            return TIFFClientOpen(
                            "internal",
                            "r",
                            src,
                            tiff_read,
                            tiff_no_write,
                            tiff_seek<byte_source>,
                            tiff_close,
                            tiff_size<byte_source>,
                            NULL,
                            NULL);
        }
        
        const int UIC1Tag = 33628;
        const int UIC2Tag = 33629;
        const int UIC3Tag = 33630;
        const int UIC4Tag = 33631;
        
        const TIFFFieldInfo stkTags[] = {
            { UIC1Tag, -1,-1, TIFF_LONG, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC1Tag") },
            { UIC1Tag, -1,-1, TIFF_RATIONAL, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC1Tag") },
            //{ UIC2Tag, -1, -1, TIFF_RATIONAL, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC2Tag") },
            { UIC2Tag, -1, -1, TIFF_LONG, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC2Tag") },
            { UIC3Tag, -1,-1, TIFF_RATIONAL, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC3Tag") },
            { UIC4Tag, -1,-1, TIFF_LONG, FIELD_CUSTOM, true, true,   const_cast<char*>("UIC4Tag") },
        };
        
        void set_stk_tags(TIFF *tif) {
            TIFFMergeFieldInfo(tif, stkTags, sizeof(stkTags)/sizeof(stkTags[0]));
        }
        
        class shift_source : public byte_source {
            public:
                explicit shift_source(byte_source *s)
                    :s(s), shift_(0)
                    {}
                
                virtual std::size_t read(byte *buf, std::size_t n) { return s->read(buf, n); }
                virtual std::size_t seek_absolute(std::size_t pos) { return s->seek_absolute(pos + shift_)-shift_; }
                virtual std::size_t seek_relative(int n) { return s->seek_relative(n)-shift_; }
                virtual std::size_t seek_end(int n) { return s->seek_end(n+shift_)-shift_; }
                
                void shift(int nshift) {
                    s->seek_relative(nshift - shift_);
                    shift_ = nshift;
                }
                
                byte_source *s;
                int shift_;
        };
        
        struct stk_extend {
            stk_extend()
                :proc(TIFFSetTagExtender(set_stk_tags))
                {}
            ~stk_extend() {
                TIFFSetTagExtender(proc);
            }
            TIFFExtendProc proc;
        };
        
    } /// namespace
    
    std::unique_ptr<ImageList> STKFormat::read_multi(byte_source *src,
                                                      ImageFactory *factory,
                                                      const options_map &opts)  {
        shift_source moved(src);
        stk_extend ext;
        tiff_warn_error twe;
        
        tif_holder t = read_client(&moved);
        std::unique_ptr<ImageList> images(new ImageList);
        const uint32_t h = tiff_get<uint32_t>(t, TIFFTAG_IMAGELENGTH);
        const uint32_t w = tiff_get<uint32_t>(t, TIFFTAG_IMAGEWIDTH);

        const uint16_t nr_samples = tiff_get<uint16_t>(t, TIFFTAG_SAMPLESPERPIXEL, 1);
        const uint16_t bits_per_sample = tiff_get<uint16_t>(t, TIFFTAG_BITSPERSAMPLE, 8);
        const int depth = nr_samples > 1 ? nr_samples : -1;

        const int strip_size = TIFFStripSize(t.tif);
        const int n_strips = TIFFNumberOfStrips(t.tif);
        int32_t n_planes;
        void *data;
        
        TIFFGetField(t.tif, UIC3Tag, &n_planes, &data);
        int raw_strip_size = 0;
        
        for (int st = 0; st != n_strips; ++st) {
            raw_strip_size += TIFFRawStripSize(t.tif, st);
        }
        
        for (int z = 0; z < n_planes; ++z) {
            /// Monkey patch strip offsets -- 
            /// This is very hacky, but it seems to work!
            moved.shift(z * raw_strip_size);
            std::unique_ptr<Image> output(factory->create(bits_per_sample, h, w, depth));
            byte *start = output->rowp_as<byte>(0);
            
            for (int st = 0; st != n_strips; ++st) {
                const int offset = TIFFReadEncodedStrip(t.tif, st, start, strip_size);
                if (offset == -1) {
                    imread_raise(CannotReadError, "Error reading strip");
                }
                start += offset;
            }
            images->push_back(std::move(output));
        }
        return images;
    }
    
    std::unique_ptr<ImageList> TIFFFormat::do_read(byte_source *src,
                                                    ImageFactory *factory,
                                                    bool is_multi)  {
        tiff_warn_error twe;
        tif_holder t = read_client(src);
        std::unique_ptr<ImageList> images(new ImageList);
        do {
            const uint32_t h = tiff_get<uint32_t>(t, TIFFTAG_IMAGELENGTH);
            const uint32_t w = tiff_get<uint32_t>(t, TIFFTAG_IMAGEWIDTH);
            const uint16_t nr_samples = tiff_get<uint16_t>(t, TIFFTAG_SAMPLESPERPIXEL);
            const uint16_t bits_per_sample = tiff_get<uint16_t>(t, TIFFTAG_BITSPERSAMPLE);
            const int depth = nr_samples > 1 ? nr_samples : -1;
            
            std::unique_ptr<Image> output = factory->create(bits_per_sample, h, w, depth);
            
            // if (ImageWithMetadata *metaout = dynamic_cast<ImageWithMetadata*>(output.get())) {
            //     std::string description = tiff_get<std::string>(t, TIFFTAG_IMAGEDESCRIPTION, "");
            //     metaout->set_meta(description);
            // }
            
            /// Hardcoding uint8_t as the type for now
            int c_stride = (depth == 1) ? 0 : output->stride(2);
            uint8_t *ptr = static_cast<uint8_t*>(output->rowp_as<uint8_t>(0));
            
            for (uint32_t r = 0; r != h; ++r) {
                byte *srcPtr = output->rowp_as<byte>(r);
                if (TIFFReadScanline(t.tif, srcPtr, r) == -1) {
                    imread_raise(CannotReadError, "Error reading scanline");
                }
                for (int x = 0; x < w; x++) {
                    for (int c = 0; c < depth; c++) {
                        /// theoretically you would want to scale this next bit,
                        /// depending on whatever the bit depth may be
                        /// -- SOMEDAAAAAAAAAAAAAAY.....
                        pix::convert(*srcPtr++, ptr[c*c_stride]);
                    }
                    ptr++;
                }
            }
            
            images->push_back(std::move(output));
        
        } while (is_multi && TIFFReadDirectory(t.tif));
        
        return images;
    }
    
    void TIFFFormat::write(Image &input, byte_sink *output, const options_map &opts)  {
        tiff_warn_error twe;
        tif_holder t = TIFFClientOpen(
                        "internal",
                        "w",
                        output,
                        tiff_no_read,
                        tiff_write,
                        tiff_seek<byte_sink>,
                        tiff_close,
                        tiff_size<byte_sink>,
                        NULL,
                        NULL);
        std::vector<byte> bufdata;
        void *bufp = 0;
        bool copy_data = false;
        const uint32_t h = input.dim(0);
        const uint16_t photometric = ((input.ndims() == 3 && input.dim(2)) ?
                                                        PHOTOMETRIC_RGB :
                                                        PHOTOMETRIC_MINISBLACK);
        
        TIFFSetField(t.tif, TIFFTAG_IMAGELENGTH, uint32_t(h));
        TIFFSetField(t.tif, TIFFTAG_IMAGEWIDTH, uint32_t(input.dim(1)));
        
        TIFFSetField(t.tif, TIFFTAG_BITSPERSAMPLE, uint16_t(input.nbits()));
        TIFFSetField(t.tif, TIFFTAG_SAMPLESPERPIXEL, uint16_t(input.dim_or(2, 1)));
        
        TIFFSetField(t.tif, TIFFTAG_PHOTOMETRIC, uint16_t(photometric));
        TIFFSetField(t.tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        
        if (get_optional_bool(opts, "tiff:compress", true)) {
            TIFFSetField(t.tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
            // For 8 bit images, prediction defaults to false; for 16 bit images,
            // it defaults to true. This is because compression of raw 16 bit
            // images is often counter-productive without this flag. See the
            // discusssion at http://www.asmail.be/msg0055176395.html
            const bool prediction_default = input.nbits() != 8;
            if (get_optional_bool(opts, "tiff:horizontal-predictor", prediction_default)) {
                TIFFSetField(t.tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
                if (!copy_data) {
                    bufdata.resize(input.dim(1) * input.nbytes());
                    bufp = &bufdata[0];
                    copy_data = true;
                }
            }
        }
        
        TIFFSetField(t.tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        const char *meta = get_optional_cstring(opts, "metadata");
        if (meta) { TIFFSetField(t.tif, TIFFTAG_IMAGEDESCRIPTION, meta); }
        if (opts.has("tiff:XResolution")) {
            TIFFSetField(t.tif, TIFFTAG_XRESOLUTION,
                opts.cast<int>("tiff:XResolution"));
        }
        if (opts.has("tiff:YResolution")) {
            TIFFSetField(t.tif, TIFFTAG_YRESOLUTION,
                opts.cast<int>("tiff:YResolution"));
        }
        if (opts.has("tiff:XResolutionUnit")) {
            TIFFSetField(t.tif, TIFFTAG_RESOLUTIONUNIT,
                opts.cast<int>("tiff:XResolutionUnit"));
        }
        
        for (uint32_t r = 0; r != h; ++r) {
            void *rowp = input.rowp(r);
            if (copy_data) {
                std::memcpy(bufp, rowp, input.dim(1) * input.nbytes());
                rowp = bufp;
            }
            if (TIFFWriteScanline(t.tif, rowp, r) == -1) {
                imread_raise(CannotWriteError, "Error writing TIFF file");
            }
        }
        TIFFFlush(t.tif);
    }
    
}