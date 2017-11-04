/// Copyright 2016 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGELIST_HH_
#define LIBIMREAD_IMAGELIST_HH_

#include <tuple>
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
    
    /// The ImageList class *owns* the pointers it manages,
    /// and it will delete them when it is destroyed
    struct ImageList {
        
        using pointer_t      = std::add_pointer_t<Image>;
        using unique_t       = std::unique_ptr<Image>;
        using vector_t       = std::vector<pointer_t>;
        using sizevec_t      = std::vector<int>;
        // using sizes_t        = std::tuple<int, int, int>;
        using pointerlist_t  = std::initializer_list<pointer_t>;
        using vector_size_t  = vector_t::size_type;
        using iterator       = vector_t::iterator;
        using const_iterator = vector_t::const_iterator;
        
        /// default constructor
        ImageList() noexcept = default;
        // ImageList();
        
        /// construct from multiple arguments
        /// ... using boolean tag for first arg
        template <typename ...Pointers>
        explicit ImageList(bool pointerargs, Pointers ...pointers)
            :images{ pointers... }
            {
                images.erase(std::remove_if(images.begin(),
                                            images.end(),
                                         [](pointer_t p) { return p == nullptr; }),
                                            images.end());
                compute_sizes();
            }
        
        /// initializer list construction
        explicit ImageList(pointerlist_t);
        
        /// move-construct from pointer vector
        explicit ImageList(vector_t&&);
        
        /// noexcept move constructor
        ImageList(ImageList&&) noexcept;
        
        /// noexcept move assignment operator
        ImageList& operator=(ImageList&&) noexcept;
        
        /// “list literal” initializer-list assignment operator
        // ImageList& operator=(pointerlist_t);
        
        /// virtual destructor
        virtual ~ImageList();
        
        vector_size_t size() const;
        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        
        void erase(iterator);
        void prepend(pointer_t);
        void push_front(pointer_t);
        void append(pointer_t);
        void push_back(pointer_t);
        void push_back(unique_t);
        
        /// calculate and cache the width/height/planecount
        /// dimensions, for the lists’ managed images
        void compute_sizes() const;
        
        /// computed-dimension-value accessors
        int width() const;
        int height() const;
        int planes() const;
        
        pointer_t get(vector_size_t) const;
        pointer_t at(vector_size_t) const;
        unique_t yank(vector_size_t);
        unique_t pop();
        void reset();
        void reset(vector_t&&);
        
        /// After calling release(), ownership of the content image ponters
        /// is transferred to the caller, who must figure out how to delete them.
        /// Also note that release() resets the internal vector.
        vector_t release();
        vector_t release(vector_t&&);
        
        /// noexcept member swap
        void swap(ImageList&) noexcept;
        
        /// member hash method
        std::size_t hash(std::size_t seed = 0) const noexcept;
        
        private:
            /// copy construct/assign are invalid
            ImageList(ImageList const&);
            ImageList& operator=(ImageList const&);
        
        protected:
            /// macro-defined dimensional compute member functions
            int compute_width() const;
            int compute_height() const;
            int compute_planes() const;
        
        protected:
            /// internal pointer vector
            vector_t images;
        
        protected:
            /// computed dimension values
            mutable int computed_width = -1,
                       computed_height = -1,
                       computed_planes = -1;
    };
    
} /* namespace im */

namespace std {
    
    template <>
    void swap(im::ImageList&, im::ImageList&) noexcept;
    
    /// std::hash specialization for im::ImageList
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::ImageList> {
        
        typedef im::ImageList argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& list) const;
        
    };
    
}; /* namespace std */


#endif /// LIBIMREAD_IMAGELIST_HH_