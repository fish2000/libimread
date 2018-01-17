/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <iterator>

#include <libimread/buffered_io.hh>

namespace im {
    
    vector_source_sink::vector_source_sink() noexcept
        :vector_ptr{ new bytevec_t }
        ,mutex_ptr{ new mutex_t }
        ,iterator{ vector_ptr->begin() }
        ,deallocate{ true }
        {}
    
    vector_source_sink::vector_source_sink(bytevec_t* bytevec_ptr)
        :vector_ptr{ bytevec_ptr }
        ,mutex_ptr{ new mutex_t }
        ,iterator{ vector_ptr->end() }
        {}
    
    vector_source_sink::vector_source_sink(bytevec_t const& bytevec)
        :vector_ptr{ new bytevec_t(bytevec) }
        ,mutex_ptr{ new mutex_t }
        ,iterator{ vector_ptr->end() }
        ,deallocate{ true }
        {}
    
    vector_source_sink::vector_source_sink(bytevec_t&& bytevec)
        :vector_ptr{ new bytevec_t(std::move(bytevec)) }
        ,mutex_ptr{ new mutex_t }
        ,iterator{ vector_ptr->end() }
        ,deallocate{ true }
        {}
    
    vector_source_sink::vector_source_sink(vector_source_sink const& other)
        :vector_ptr{ new bytevec_t(*other.vector_ptr) }
        ,mutex_ptr{ new mutex_t }
        ,iterator{ vector_ptr->end() }
        ,deallocate{ true }
        {}
    
    vector_source_sink::vector_source_sink(vector_source_sink&& other) noexcept
        :vector_ptr{ std::exchange(other.vector_ptr, nullptr) }
        ,mutex_ptr{ new mutex_t }
        ,iterator{ vector_ptr->end() }
        ,deallocate{ other.deallocate }
        {
            other.deallocate = false;
        }
    
    vector_source_sink::~vector_source_sink() {
        if (deallocate) {
            std::lock_guard<std::mutex>(*mutex_ptr);
            delete vector_ptr;
        }
        delete mutex_ptr;
    }
    
    bool vector_source_sink::can_seek() const noexcept { return true; }
    
    std::size_t vector_source_sink::seek_absolute(std::size_t pos) {
        if (pos < vector_ptr->size()) {
            iterator = vector_ptr->begin();
            std::advance(iterator, pos);
            return std::distance(vector_ptr->begin(), iterator);
        } else if (pos == vector_ptr->size()) {
            iterator = vector_ptr->end();
            return vector_ptr->size();
        }
        return 0;
    }
    
    std::size_t vector_source_sink::seek_relative(int delta) {
        int pos = std::distance(vector_ptr->begin(), iterator) + delta;
        if (pos < vector_ptr->size()) {
            std::advance(iterator, delta);
            return std::distance(vector_ptr->begin(), iterator);
        } else if (pos == vector_ptr->size()) {
            iterator = vector_ptr->end();
            return vector_ptr->size();
        }
        return 0;
    }
    
    std::size_t vector_source_sink::seek_end(int delta) {
        int pos = (vector_ptr->size() - delta - 1);
        if (pos < vector_ptr->size()) {
            iterator = vector_ptr->begin();
            std::advance(iterator, pos);
            return std::distance(vector_ptr->begin(), iterator);
        } else if (pos == vector_ptr->size()) {
            iterator = vector_ptr->end();
            return vector_ptr->size();
        }
        return 0;
    }
    
    std::size_t vector_source_sink::read(byte* buffer, std::size_t nbytes) const {
        {
            std::lock_guard<std::mutex>(*mutex_ptr);
            byte* source = vector_ptr->data() + std::distance(vector_ptr->begin(), iterator);
            std::memcpy(buffer, source, nbytes);
            std::advance(iterator, nbytes);
        }
        return nbytes;
    }
    
    vector_source_sink::bytevec_t vector_source_sink::full_data() const {
        return bytevec_t(*vector_ptr);
    }
    
    std::size_t vector_source_sink::size() const {
        return vector_ptr->size();
    }
    
    std::size_t vector_source_sink::write(const void* buffer, std::size_t nbytes) {
        {
            const byte* bytes = static_cast<const byte*>(buffer);
            std::lock_guard<std::mutex>(*mutex_ptr);
            vector_ptr->reserve(vector_ptr->capacity() + nbytes);
            iterator = vector_ptr->insert(iterator,
                                          bytes,
                                          bytes + nbytes);
        }
        return nbytes;
    }
    
    std::size_t vector_source_sink::write(bytevec_t const& bytevec) {
        {
            std::lock_guard<std::mutex>(*mutex_ptr);
            vector_ptr->reserve(vector_ptr->capacity() + bytevec.size());
            iterator = vector_ptr->insert(iterator,
                                          std::begin(bytevec),
                                          std::end(bytevec));
        }
        return bytevec.size();
    }
    
    std::size_t vector_source_sink::write(bytevec_t&& bytevec) {
        {
            std::lock_guard<std::mutex>(*mutex_ptr);
            vector_ptr->reserve(vector_ptr->capacity() + bytevec.size());
            iterator = vector_ptr->insert(iterator,
                                          std::begin(bytevec),
                                          std::end(bytevec));
        }
        return bytevec.size();
    }
    
    void vector_source_sink::flush() {}
    
    void* vector_source_sink::readmap(std::size_t pageoffset) const {
        byte* out = vector_ptr->data();
        if (pageoffset) {
            out += pageoffset * ::getpagesize();
        }
        return static_cast<void*>(out);
    }
    
    void vector_source_sink::reserve(std::size_t new_cap) {
        vector_ptr->reserve(new_cap);
    }
    
    void vector_source_sink::push_back(byte const& value) {
        vector_ptr->reserve(vector_ptr->capacity() + 1);
        vector_ptr->emplace_back(value);
        iterator = vector_ptr->end();
    }
    
    void vector_source_sink::push_back(byte&& value) {
        vector_ptr->reserve(vector_ptr->capacity() + 1);
        vector_ptr->emplace_back(value);
        iterator = vector_ptr->end();
    }
    
    void vector_source_sink::push_front(byte const& value) {
        vector_ptr->reserve(vector_ptr->capacity() + 1);
        vector_ptr->emplace(vector_ptr->begin(), value);
        iterator = vector_ptr->begin();
    }
    
    void vector_source_sink::push_front(byte&& value) {
        vector_ptr->reserve(vector_ptr->capacity() + 1);
        vector_ptr->emplace(vector_ptr->begin(), value);
        iterator = vector_ptr->begin();
    }
    
    
} /// namespace im