/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_BUFFERED_IO_HH_
#define LIBIMREAD_BUFFERED_IO_HH_

#include <iostream>
#include <vector>
#include <mutex>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>

namespace im {
    
    class vector_source_sink : public byte_source, public byte_sink {
        
        protected:
            using byte_sink::kBufferSize;
        
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
            using vector_type               =          bytevec_t;
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
            void clear() noexcept;
        
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
    
    template <typename PrimarySink, typename AlternateSink>
    class modal_sink : public byte_sink {
        
        public:
            using primary_sink_t = PrimarySink;
            using alternate_sink_t = AlternateSink;
        
        public:
            using byte_sink::value_type;
            using byte_sink::difference_type;
            using byte_sink::size_type;
            using byte_sink::reference_type;
            using byte_sink::reference;
            using byte_sink::const_reference;
            using byte_sink::pointer;
            using byte_sink::vector_type;
            using byte_sink::iterator;
            using byte_sink::const_iterator;
            using byte_sink::reverse_iterator;
            using byte_sink::const_reverse_iterator;
        
        public:
            explicit modal_sink(primary_sink_t* primary_sink_ptr)
                :primary{ primary_sink_ptr }
                ,alternate{ primary_sink_ptr }
                ,current{ primary }
                {}
            
            explicit modal_sink(primary_sink_t* primary_sink_ptr,
                                alternate_sink_t* alternate_sink_ptr)
                :primary{ primary_sink_ptr }
                ,alternate{ alternate_sink_ptr }
                ,current{ primary }
                {}
        
        public:
            virtual ~modal_sink() {
                std::cerr << "modal_sink destructor" << std::endl;
            }
        
        public:
            /// im::seekable methods
            virtual bool can_seek() const noexcept override                 { return current->can_seek(); }
            virtual std::size_t seek_absolute(std::size_t pos) override     { return current->seek_absolute(pos); }
            virtual std::size_t seek_relative(int delta) override           { return current->seek_relative(delta); }
            virtual std::size_t seek_end(int delta) override                { return current->seek_end(delta); }
        
        public:
            /// im::byte_sink methods
            virtual std::size_t write(const void* buffer, std::size_t nbytes) override  { return current->write(buffer, nbytes); }
            virtual std::size_t write(bytevec_t const& bytevec) override                { return current->write(bytevec); }
            virtual std::size_t write(bytevec_t&& bytevec) override                     { return current->write(std::forward<bytevec_t>(bytevec)); }
            virtual void flush() override                                               { current->flush(); }
        
        public:
            void push_back(byte const& value)                                           { current->push_back(value); }
            void push_back(byte&& value)                                                { current->push_back(value); }
            
        public:
            void push_front(byte const& value)                                          { current->push_front(value); }
            void push_front(byte&& value)                                               { current->push_front(value); }
        
        public:
            primary_sink_t* get_primary() const                                         { return primary; }
            alternate_sink_t* get_alternate() const                                     { return alternate; }
            void set_alternate(alternate_sink_t* alternate_sink_ptr)                    { alternate = alternate_sink_ptr; }
            void enable_primary() const                                                 { current = dynamic_cast<byte_sink*>(primary); }
            void enable_alternate() const                                               { current = dynamic_cast<byte_sink*>(alternate); }
            void enable_alternate(alternate_sink_t* alternate_sink_ptr) const           { alternate = alternate_sink_ptr;
                                                                                          current = dynamic_cast<byte_sink*>(alternate); }
        
        protected:
            mutable PrimarySink*    primary     = nullptr;
            mutable AlternateSink*  alternate   = nullptr;
            mutable byte_sink*      current     = nullptr;
    };
    
    template <typename Sink>
    class buffered_sink : public modal_sink<Sink, vector_source_sink> {
        
        public:
            using sink_t = Sink;
            using sink_buffer_t = vector_source_sink;
            using modal_t = modal_sink<Sink, vector_source_sink>;
        
        public:
            using modal_t::modal_t;
        
        public:
            using typename modal_t::primary_sink_t;
            using typename modal_t::alternate_sink_t;
            using modal_t::primary;
            using modal_t::alternate;
            using modal_t::current;
            using modal_t::get_primary;
            using modal_t::get_alternate;
            using modal_t::set_alternate;
            using modal_t::enable_primary;
            using modal_t::enable_alternate;
        
        public:
            using typename modal_t::value_type;
            using typename modal_t::difference_type;
            using typename modal_t::size_type;
            using typename modal_t::reference_type;
            using typename modal_t::reference;
            using typename modal_t::const_reference;
            using typename modal_t::pointer;
            using typename modal_t::vector_type;
            using typename modal_t::iterator;
            using typename modal_t::const_iterator;
            using typename modal_t::reverse_iterator;
            using typename modal_t::const_reverse_iterator;
        
        public:
            buffered_sink(sink_t* sink_ptr)
                :modal_t(sink_ptr, new sink_buffer_t)
                ,sink{ modal_t::get_primary() }
                ,sink_buffer{ modal_t::get_alternate() }
                {
                    modal_t::enable_alternate();
                }
            
        public:
            virtual ~buffered_sink() {
                std::cerr << "buffered_sink destructor" << std::endl;
            }
            
        public:
            virtual void flush() override {
                // sink_buffer_t* vecbuf = dynamic_cast<sink_buffer_t*>(alternate);
                // sink_t* target = dynamic_cast<sink_t*>(primary);
                // sink_buffer_t* vecbuf = dynamic_cast<sink_buffer_t*>(sink_buffer.get());
                // sink_t* target = dynamic_cast<sink_t*>(sink.get());
                // sink_buffer_t* vecbuf = sink_buffer.get();
                // sink_t* target = sink.get();
                std::size_t siz = sink_buffer->size();
                if (siz < 1) { return; }
                // __attribute__((unused))
                // std::size_t written = target->write((const void*)vecbuf->data(), siz);
                // if (written != siz) {} /// RAISE?!?!
                // sink->write((const void*)vecbuf->data(), siz);
                sink->write(sink_buffer->full_data());
                // sink->flush();
                sink_buffer->clear();
            }
            
        public:
            void enable_buffering()  const { enable_alternate(); }
            void disable_buffering() const { enable_primary(); }
            bool buffering_enabled() const { return current == alternate; }
            
        protected:
            std::unique_ptr<sink_t>         sink{ nullptr };            /// WILL BE DELETED!
            std::unique_ptr<sink_buffer_t>  sink_buffer{ nullptr };     /// WILL ALSO BE DELETED!
    };
    
} /// namespace im


#endif /// LIBIMREAD_BUFFERED_IO_HH_