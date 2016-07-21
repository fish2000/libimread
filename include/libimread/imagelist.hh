/// Copyright 2016 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGELIST_HH_
#define LIBIMREAD_IMAGELIST_HH_

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <initializer_list>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    /// forward-declare im::Image
    class Image;
    
    /// This class *owns* its members and will delete them if destroyed
    struct ImageList {
        using pointer_t      = std::add_pointer_t<Image>;
        using unique_t       = std::unique_ptr<Image>;
        using vector_t       = std::vector<pointer_t>;
        using pointerlist_t  = std::initializer_list<pointer_t>;
        using vector_size_t  = vector_t::size_type;
        using iterator       = vector_t::iterator;
        using const_iterator = vector_t::const_iterator;
        
        /// default constructor
        ImageList() noexcept = default;
        
        /// construct from multiple arguments
        /// ... using boolean tag for first arg
        template <typename ...Pointers>
        explicit ImageList(bool pointerargs, Pointers ...pointers)
            :content{ pointers... }
            {
                content.erase(
                    std::remove_if(content.begin(), content.end(),
                                [](pointer_t p) { return p == nullptr; }),
                    content.end());
            }
        
        /// initializer list construction
        explicit ImageList(pointerlist_t pointerlist);
        
        /// move-construct from pointer vector
        explicit ImageList(vector_t&& vector);
        
        /// noexcept move constructor
        ImageList(ImageList&& other) noexcept;
        
        /// noexcept move assignment operator
        ImageList& operator=(ImageList&& other) noexcept;
        
        vector_size_t size() const;
        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        
        void erase(iterator it);
        void prepend(pointer_t image);
        void push_front(pointer_t image);
        void append(pointer_t image);
        void push_back(pointer_t image);
        void push_back(unique_t unique);
        
        pointer_t get(vector_size_t idx) const;
        pointer_t at(vector_size_t idx) const;
        unique_t yank(vector_size_t idx);
        unique_t pop();
        void reset();
        void reset(vector_t&& vector);
        virtual ~ImageList();
        
        /// After calling release(), ownership of the content image ponters
        /// is transferred to the caller, who must figure out how to delete them.
        /// Also note that release() resets the internal vector.
        vector_t release();
        
        /// noexcept member swap
        void swap(ImageList& other) noexcept;
        
        /// member hash method
        std::size_t hash(std::size_t seed = 0) const noexcept;
        
        private:
            ImageList(const ImageList&);
            ImageList &operator=(const ImageList&);
            vector_t content;
    };
    
} /* namespace im */

namespace std {
    
    template <>
    void swap(im::ImageList& p0, im::ImageList& p1) noexcept;
    
    /// std::hash specialization for im::ImageList
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::ImageList> {
        
        typedef im::ImageList argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& list) const {
            return static_cast<result_type>(list.hash());
        }
        
    };
    
}; /* namespace std */


#endif /// LIBIMREAD_IMAGELIST_HH_