// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <inttypes.h>
#include <cassert>
#include <type_traits>
#include <memory>
#include <vector>
#include <string>
#include <cstring>

#if defined(_MSC_VER)
 #include <io.h>
 #include <fcntl.h>
#else
 #include <unistd.h>
#endif

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/options.hh>

namespace im {
    
    struct seekable {
        virtual ~seekable() { }
        virtual bool can_seek() const { return false; }
        virtual std::size_t seek_absolute(std::size_t) { throw NotImplementedError(); }
        virtual std::size_t seek_relative(int) { throw NotImplementedError(); }
        virtual std::size_t seek_end(int) { throw NotImplementedError(); }
    };

    class byte_source : virtual public seekable {
        
        public:
            virtual ~byte_source() { }
            virtual std::size_t read(byte *buffer, std::size_t) xWARN_UNUSED = 0;
            
            template <std::size_t Nelems>
            std::size_t read(byte (&arr)[Nelems], size_t n) {
                assert(n <= Nelems);
                byte *p = arr;
                return this->read(p, n);
            }
            
            void read_check(byte *buffer, std::size_t n) {
                if (this->read(buffer, n) != n) {
                    throw CannotReadError("File ended prematurely");
                }
            }
            
            virtual std::vector<byte> full_data() {
                std::vector<byte> res;
                byte buffer[4096];
                while (int n = this->read(buffer, sizeof(buffer))) {
                    res.insert(res.end(), buffer, buffer + n);
                }
                return res;
            }
            
    };

    class byte_sink : virtual public seekable {
        public:
            virtual ~byte_sink() { }
            virtual std::size_t write(const byte *buffer, std::size_t n) xWARN_UNUSED = 0;
            
            template <std::size_t Nelems>
            std::size_t write(byte (&arr)[Nelems], size_t n) {
                assert(n <= Nelems);
                byte *p = arr;
                return this->write(p, n);
            }
            
            void write_check(const byte *buffer, std::size_t n) {
                if (this->write(buffer, n) != n) {
                    throw CannotWriteError("Writing failed");
                }
            }
            virtual void flush() { }
    };

    class Image {
        public:
            virtual ~Image() { }
            
            virtual void *rowp(int r) = 0;
            virtual int nbits() const = 0;
            
            virtual int nbytes() const {
                const int bits = this->nbits();
                return (bits / 8) + bool(bits % 8);
            }
            
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            
            virtual int dim_or(int dim, int def) const {
                if (dim >= this->ndims()) { return def; }
                return this->dim(dim);
            }
            
            template<typename T>
            T *rowp_as(const int r) {
                return static_cast<T*>(this->rowp(r));
            }
            
            template <typename T>
            inline std::vector<T*> allrows() {
                std::vector<T*> res;
                const int h = this->dim(0);
                for (int r = 0; r != h; ++r) {
                    res.push_back(this->rowp_as<T>(r));
                }
                return res;
            }
            
    };
    
    class ImageFactory {
        public:
            virtual ~ImageFactory() { }
            virtual std::unique_ptr<Image>
                create(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
        protected:
    };
    
    class ImageWithMetadata {
        public:
            ImageWithMetadata()
                :meta(0)
                {}
            ImageWithMetadata(const std::string &m)
                :meta(m == "" ? 0 : new std::string(m))
                {}
            
            virtual ~ImageWithMetadata() { delete meta; };
            std::string *get_meta() { return meta; }
            
            void set_meta(const std::string &m) {
                if (meta) { delete meta; }
                meta = new std::string(m);
            }
            
        private:
            std::string *meta;
    };
    
    /// This class *owns* its members and will delete them if destroyed
    struct image_list {
        public:
            image_list() { }
            ~image_list() {
                for (unsigned i = 0; i != content.size(); ++i) {
                    delete content[i];
                }
            }
            
            std::vector<Image*>::size_type size() const { return content.size(); }
            void push_back(std::unique_ptr<Image> p) { content.push_back(p.release()); }
            
            /// After release(), all of the pointers will be owned by the caller
            /// who must figure out how to delete them. Note that release() resets the list.
            std::vector<Image*> release() {
                std::vector<Image*> r;
                r.swap(content);
                return r;
            }
        private:
            image_list(const image_list&);
            image_list &operator=(const image_list&);
            std::vector<Image*> content;
    };
    
    inline bool match_magic(byte_source *src, const char *magic, const std::size_t n) {
        if (!src->can_seek()) { return false; }
        std::vector<byte> buf;
        buf.resize(n);
        const std::size_t n_read = src->read(&buf.front(), n);
        src->seek_relative(-n_read);
    
        return (n_read == n && std::memcmp(&buf.front(), magic, n) == 0);
    }
    
    template <typename>
    struct sfinae_true : std::true_type{};

    namespace detail {
        template <typename T, typename ...Args>
        static auto test_read(int) -> typename T::can_read;
        template <typename, typename ...Args>
        static auto test_read(long) -> std::false_type;
        
        template <typename T, typename ...Args>
        static auto test_write(int) -> typename T::can_write;
        template <typename, typename ...Args>
        static auto test_write(long) -> std::false_type;
        
        template <typename T, typename ...Args>
        static auto test_read_multi(int) -> typename T::can_read_multi;
        template <typename, typename ...Args>
        static auto test_read_multi(long) -> std::false_type;
        
        template <typename T, typename ...Args>
        static auto test_write_metadata(int) -> typename T::can_write_metadata;
        template <typename, typename ...Args>
        static auto test_write_metadata(long) -> std::false_type;
    }
    
#define TEST_ARGS byte_source*, ImageFactory*, const options_map&
    
    template <typename T>
    struct has_read : decltype(detail::test_read<T, TEST_ARGS>(0)) {
        template <typename X = std::enable_if<decltype(detail::test_read<T, TEST_ARGS>(0))::value>>
        static constexpr bool value() { return true; }
        static constexpr bool value() { return detail::test_read<T, TEST_ARGS>(0); }
    };
    
    template <typename T>
    struct has_write : decltype(detail::test_write<T, TEST_ARGS>(0)){
        template <typename X = std::enable_if<decltype(detail::test_write<T, TEST_ARGS>(0))::value>>
        static constexpr bool value() { return true; }
        static constexpr bool value() { return detail::test_write<T, TEST_ARGS>(0); }
    };
    
    template <typename T>
    struct has_read_multi : decltype(detail::test_read_multi<T, TEST_ARGS>(0)){
        template <typename X = std::enable_if<decltype(detail::test_read_multi<T, TEST_ARGS>(0))::value>>
        static constexpr bool value() { return true; }
        static constexpr bool value() { return detail::test_read_multi<T, TEST_ARGS>(0); }
    };
    
    template <typename T>
    struct has_write_metadata : decltype(detail::test_write_metadata<T, TEST_ARGS>(0)){
        template <typename X = std::enable_if<decltype(detail::test_write_metadata<T, TEST_ARGS>(0))::value>>
        static constexpr bool value() { return true; }
        static constexpr bool value() { return detail::test_write_metadata<T, TEST_ARGS>(0); }
    };
    
#undef TEST_ARGS
    
    class ImageFormat {
        public:
            typedef std::false_type can_read;
            typedef std::false_type can_read_multi;
            typedef std::false_type can_write;
            typedef std::false_type can_read_metadata;
            
            virtual ~ImageFormat() {}
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) {
                throw NotImplementedError();
            }
            
            virtual std::unique_ptr<image_list> read_multi(byte_source *src,
                                                           ImageFactory *factory,
                                                           const options_map &opts) {
                throw NotImplementedError();
            }
            
            virtual void write(Image *input,
                               byte_sink *output,
                               const options_map &opts) {
                throw NotImplementedError();
            }
            
    };

}

#endif // LPC_IMREAD_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
