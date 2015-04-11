// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <libimread/IO/jpeg.hh>

extern "C" {
    #include <jpeglib.h>
}

namespace im {

    namespace {
        
        const std::size_t buffer_size = 4096;
        
        struct jpeg_source_adaptor {
            jpeg_source_mgr mgr;
            byte_source *s;
            byte *buf;
            
            jpeg_source_adaptor(byte_source *s);
            ~jpeg_source_adaptor() { delete[] buf; }
        };
        struct jpeg_dst_adaptor {
            jpeg_destination_mgr mgr;
            byte_sink *s;
            byte *buf;
            
            jpeg_dst_adaptor(byte_sink *s);
            ~jpeg_dst_adaptor() { delete[] buf; }
        };
        
        void nop(j_decompress_ptr cinfo) {}
        void nop_dst(j_compress_ptr cinfo) {}
        
        boolean fill_input_buffer(j_decompress_ptr cinfo) {
            jpeg_source_adaptor *adaptor
                = reinterpret_cast<jpeg_source_adaptor *>(cinfo->src);
            adaptor->mgr.next_input_byte = adaptor->buf;
            adaptor->mgr.bytes_in_buffer
                = adaptor->s->read(adaptor->buf, buffer_size);
            return true;
        }
        
        void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
            if (num_bytes <= 0) { return; }
            jpeg_source_adaptor *adaptor = reinterpret_cast<jpeg_source_adaptor *>(cinfo->src);
            while (num_bytes > long(adaptor->mgr.bytes_in_buffer)) {
                num_bytes -= adaptor->mgr.bytes_in_buffer;
                fill_input_buffer(cinfo);
            }
            adaptor->mgr.next_input_byte += num_bytes;
            adaptor->mgr.bytes_in_buffer -= num_bytes;
        }
        
        boolean empty_output_buffer(j_compress_ptr cinfo) {
            jpeg_dst_adaptor *adaptor = reinterpret_cast<jpeg_dst_adaptor *>(cinfo->dest);
            adaptor->s->write_check(
                adaptor->buf,
                buffer_size);
            adaptor->mgr.next_output_byte = adaptor->buf;
            adaptor->mgr.free_in_buffer = buffer_size;
            return TRUE;
        }
        
        void flush_output_buffer(j_compress_ptr cinfo) {
            jpeg_dst_adaptor *adaptor = reinterpret_cast<jpeg_dst_adaptor *>(cinfo->dest);
            adaptor->s->write_check(
                adaptor->buf,
                adaptor->mgr.next_output_byte - adaptor->buf);
        }
        
        jpeg_source_adaptor::jpeg_source_adaptor(byte_source *s)
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
        
        jpeg_dst_adaptor::jpeg_dst_adaptor(byte_sink *s)
            :s(s)
            {
                buf = new byte[buffer_size];
                mgr.next_output_byte = buf;
                mgr.free_in_buffer = buffer_size;
                mgr.init_destination = nop_dst;
                mgr.empty_output_buffer = empty_output_buffer;
                mgr.term_destination = flush_output_buffer;
            }
        
        struct jpeg_decompress_holder {
            jpeg_decompress_holder() { jpeg_create_decompress(&info); }
            ~jpeg_decompress_holder() { jpeg_destroy_decompress(&info); }
            jpeg_decompress_struct info;
        };
        
        struct jpeg_compress_holder {
            jpeg_compress_holder() { jpeg_create_compress(&info); }
            ~jpeg_compress_holder() { jpeg_destroy_compress(&info); }
            jpeg_compress_struct info;
        };
        
        struct error_mgr {
            error_mgr();
            struct jpeg_error_mgr pub;
            jmp_buf setjmp_buffer;
            char error_message[JMSG_LENGTH_MAX];
        };
        
        void err_long_jump(j_common_ptr cinfo) {
            error_mgr *err = reinterpret_cast<error_mgr *>(cinfo->err);
            (*cinfo->err->format_message)(cinfo, err->error_message);
            longjmp(err->setjmp_buffer, 1);
        }
        
        error_mgr::error_mgr() {
            jpeg_std_error(&pub);
            pub.error_exit = err_long_jump;
            error_message[0] = 0;
        }
        
        inline J_COLOR_SPACE color_space(int components) {
            if (components == 1) { return JCS_GRAYSCALE; }
            if (components == 3) { return JCS_RGB; }
            std::ostringstream out;
            out << "ERROR:\n"
                << "\tim::(anon)::color_space() says:   \"UNSUPPORTED IMAGE DIMENSIONS\"\n"
                << "\tim::(anon)::color_space() got:    `components` = (int)" << components << "\n"
                << "\tim::(anon)::color_space() needs:  `components` = (int){ 1, 3 }\n";
            throw CannotReadError(out.str());
        }
        
        /// pixel accessor shortcut
        template <typename T = uint8_t>
        inline T *at(Image &im, int x, int y, int z) {
            return &im.rowp_as<T>(0)[x*im.stride(0) +
                                     y*im.stride(1) +
                                     z*im.stride(2)];
        }
        
        
    } /// namespace
    
    std::unique_ptr<Image> JPEGFormat::read(byte_source *src,
                           ImageFactory *factory,
                           const options_map &opts)  {
        
        jpeg_source_adaptor adaptor(src);
        jpeg_decompress_holder decompressor;
        
        /// error management
        error_mgr jerr;
        decompressor.info.err = &jerr.pub;
        
        /// source
        decompressor.info.src = &adaptor.mgr;
        
        if (setjmp(jerr.setjmp_buffer)) {
            throw CannotReadError(jerr.error_message);
        }
        
        // now read the header & image data
        jpeg_read_header(&decompressor.info, TRUE);
        jpeg_start_decompress(&decompressor.info);
        
        if (setjmp(jerr.setjmp_buffer)) {
            throw CannotReadError(jerr.error_message);
        }
        
        const int h = decompressor.info.output_height;
        const int w = decompressor.info.output_width;
        const int d = decompressor.info.output_components;
        
        std::unique_ptr<Image> output(factory->create(8, h, w, d));
        
        JSAMPARRAY samples = (*decompressor.info.mem->alloc_sarray)(
             (j_common_ptr)&decompressor.info, JPOOL_IMAGE, w * d, 1);
        
        /// Hardcoding uint8_t as the type for now
        int c_stride = (d == 1) ? 0 : output->stride(2);
        uint8_t *ptr = output->rowp_as<uint8_t>(0);
        
        while (decompressor.info.output_scanline < decompressor.info.output_height) {
            jpeg_read_scanlines(&decompressor.info, samples, 1);
            JSAMPLE *srcPtr = samples[0];
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < d; c++) {
                    /// theoretically you would want to scale this next bit,
                    /// depending on whatever the bit depth may be
                    /// -- SOMEDAAAAAAAAAAAAAAY.....
                    pix::convert(*srcPtr++, ptr[c*c_stride]);
                }
                ptr++;
            }
        }
        
        if (setjmp(jerr.setjmp_buffer)) {
            throw CannotReadError(jerr.error_message);
        }
        
        jpeg_finish_decompress(&decompressor.info);
        return output;
    }
    
    void JPEGFormat::write(Image &input,
                           byte_sink *output,
                           const options_map &opts) {
        if (input.nbits() != 8) {
            throw CannotWriteError("im::JPEGFormat::write(): Image must be 8 bits for JPEG saving");
        }
        
        jpeg_dst_adaptor adaptor(output);
        jpeg_compress_holder compressor;
        
        /// error management
        error_mgr jerr;
        compressor.info.err = &jerr.pub;
        
        /// destination
        compressor.info.dest = &adaptor.mgr;
        
        if (setjmp(jerr.setjmp_buffer)) { throw CannotWriteError(
            std::string("im::JPEGFormat::write(): ") + std::string(jerr.error_message)); }
        
        const int w = input.dim(0);
        const int h = input.dim(1);
        const int d = input.dim(2);
        const int dims = input.ndims();
        
        compressor.info.image_width = w;
        compressor.info.image_height = h;
        compressor.info.input_components = (dims > 2 ? input.dim(2) : 1);
        compressor.info.in_color_space = color_space(compressor.info.input_components);
        
        jpeg_set_defaults(&compressor.info);
        
        if (setjmp(jerr.setjmp_buffer)) { throw CannotWriteError(
            std::string("im::JPEGFormat::write(): ") + std::string(jerr.error_message)); }
        
        options_map::const_iterator qiter = opts.find("jpeg:quality");
        if (qiter != opts.end()) {
            int quality;
            if (qiter->second.get_int(quality)) {
                if (quality > 100) { quality = 100; }
                if (quality < 0) { quality = 0; }
                jpeg_set_quality(&compressor.info, quality, FALSE);
            } else {
                throw WriteOptionsError(
                    "im::JPEGFormat::write(): jpeg:quality must be an integer"
                );
            }
        }
        
        JSAMPLE *rowbuf = new JSAMPLE[w * d]; /// width * channels
        
        jpeg_start_compress(&compressor.info, TRUE);
        
        if (setjmp(jerr.setjmp_buffer)) { throw CannotWriteError(
            std::string("im::JPEGFormat::write(): ") + std::string(jerr.error_message)); }
        
        while (compressor.info.next_scanline < compressor.info.image_height) {
            JSAMPLE *dstPtr = rowbuf;
            for (int x = 0; x < w; x++) {
                for (int c = 0; c < d; c++) {
                    pix::convert(*dstPtr++, at<JSAMPLE>(input,
                        x, compressor.info.next_scanline, c)[0]);
                }
            }
            jpeg_write_scanlines(&compressor.info, &rowbuf, 1);
        }
        
        if (setjmp(jerr.setjmp_buffer)) { throw CannotWriteError(
            std::string("im::JPEGFormat::write(): ") + std::string(jerr.error_message)); }
        
        delete[] rowbuf;
        jpeg_finish_compress(&compressor.info);
    }
}
