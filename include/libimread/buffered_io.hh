/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_BUFFERED_IO_HH_
#define LIBIMREAD_BUFFERED_IO_HH_

#include <vector>
#include <mutex>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>

namespace im {
    
    class vector_source_sink : public byte_source, public byte_sink {
        
        public:
            using bytevec_t                 = std::vector<byte>;
            using iterator_t                = typename bytevec_t::iterator;
            using mutex_t                   = std::mutex;
        
        public:
            using value_type                = typename bytevec_t::value_type;
            using difference_type           = typename bytevec_t::difference_type;
            using size_type                 = typename bytevec_t::size_type;
            using reference_type            = typename bytevec_t::reference;
            using reference                 = typename bytevec_t::reference;
            using const_reference           = typename bytevec_t::const_reference;
            using pointer                   = typename bytevec_t::pointer;
            using iterator_type             = typename bytevec_t::iterator;
            using const_iterator            = typename bytevec_t::const_iterator;
            using reverse_iterator          = typename bytevec_t::reverse_iterator;
            using const_reverse_iterator    = typename bytevec_t::const_reverse_iterator;
        
        public:
            vector_source_sink() noexcept;
            explicit vector_source_sink(bytevec_t*);
            explicit vector_source_sink(bytevec_t const&);
            explicit vector_source_sink(bytevec_t&&);
        
        public:
            vector_source_sink(vector_source_sink const&);
            vector_source_sink(vector_source_sink&&) noexcept;
            virtual ~vector_source_sink();
        
        public:
            /// im::seekable methods
            virtual bool can_seek() const noexcept override;
            virtual std::size_t seek_absolute(std::size_t) override;
            virtual std::size_t seek_relative(int) override;
            virtual std::size_t seek_end(int) override;
            
        public:
            /// im::byte_source and im::byte_sink methods
            virtual std::size_t read(byte*, std::size_t) const override;
            virtual bytevec_t full_data() const override;
            virtual std::size_t size() const override;
            virtual std::size_t write(const void*, std::size_t) override;
            virtual std::size_t write(bytevec_t const&) override;
            virtual std::size_t write(bytevec_t&&) override;
            virtual void flush() override;
            
        public:
            virtual void* readmap(std::size_t pageoffset = 0) const override;
        
        public:
            void reserve(std::size_t);
        
        public:
            void push_back(byte const&);
            void push_back(byte&&);
            
        public:
            void push_front(byte const&);
            void push_front(byte&&);
        
        protected:
            bytevec_t*          vector_ptr  = nullptr;
            mutable mutex_t*    mutex_ptr   = nullptr;
            mutable iterator_t  iterator;
            bool                deallocate  = false;
        
    };
    
} /// namespace im


#endif /// LIBIMREAD_BUFFERED_IO_HH_