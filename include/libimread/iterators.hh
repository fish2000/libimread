/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ITERATORS_HH_
#define LIBIMREAD_ITERATORS_HH_

#include <stack>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>

namespace im {
    
    namespace detail {
        
        template <typename IteratorType>
        class iterator_base {
            
            public:
                using difference_type = std::ptrdiff_t;
                using idx_t = difference_type; /// Not into typing it out over and over
                using value_type = byte;
                using reference = std::add_lvalue_reference_t<value_type>;
                using pointer = std::add_pointer_t<value_type>;
                using iterator_category = std::forward_iterator_tag;
                
                iterator_base() {}
                
                iterator_base(iterator_base const& other)
                    :idx(other.idx)
                    {}
                
                virtual ~iterator_base() {}
                
                IteratorType& operator=(IteratorType const& other) {
                    IteratorType(other).swap(*this);
                    return *this;
                }
                
                IteratorType& operator++() {
                    ++idx;
                    return *this;
                }
                
                friend void swap(iterator_base& lhs, iterator_base& rhs) {
                    using std::swap;
                    swap(lhs.idx, rhs.idx);
                }
                
            protected:
                idx_t pushidx() {
                    stash.push(idx);
                    return idx;
                }
                
                idx_t popidx() {
                    idx_t out = stash.top();
                    stash.pop();
                    return out;
                }
                
            protected:
                idx_t idx = 0;
                std::stack<idx_t> stash;
        };
        
    }
    
    class source_iterator : public virtual detail::iterator_base<source_iterator> {
        
        using iterator_base_t = detail::iterator_base<source_iterator>;
        
        public:
            source_iterator(source_iterator const& other)
                :iterator_base_t(other)
                ,source(other.source)
                {}
            
            explicit source_iterator(byte_source* s)
                :idx(source->seek_relative(0))
                ,source(s)
                ,onebyte{ new byte[1] }
                {
                    pushidx();
                }
            
            virtual ~source_iterator() {
                source->seek_absolute(popidx());
            }
            
            value_type operator*() const {
                source->seek_absolute(idx);
                source->read(onebyte.get(), 1);
                return onebyte.get();
            }
            
            friend void swap(source_iterator& lhs, source_iterator& rhs) {
                using std::swap;
                swap(lhs.idx,    rhs.idx);
                swap(lhs.source, rhs.source);
            }
        
        private:
            byte_source* source = nullptr;
            std::unique_ptr<byte[]> onebyte;
    };
    
}


#endif /// LIBIMREAD_ITERATORS_HH_