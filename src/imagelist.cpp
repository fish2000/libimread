/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/imagelist.hh>
#include <libimread/image.hh>
#include <libimread/rehash.hh>

#define DEFINE_DIMENSION_COMPUTE_FN(__dimension__)                                  \
                                                                                    \
    int ImageList::compute_##__dimension__() const {                                \
        using sizevec_t = ImageList::sizevec_t;                                     \
        using pointer_t = ImageList::pointer_t;                                     \
        if (images.empty()) { return -1; }                                          \
        sizevec_t sizes;                                                            \
        sizes.reserve(images.size());                                               \
        std::for_each(images.begin(),                                               \
                      images.end(),                                                 \
             [&sizes](pointer_t p) { sizes.emplace_back(p->__dimension__()); });    \
        int base = images[0]->__dimension__();                                      \
        if (std::all_of(sizes.cbegin(),                                             \
                        sizes.cend(),                                               \
                    [=](int size) { return size == base; })) {                      \
            return base;                                                            \
        }                                                                           \
        return -1;                                                                  \
    }

namespace im {
    
    namespace detail {
        using rehasher_t = hash::rehasher<ImageList::pointer_t>;
    }
    
    ImageList::ImageList(ImageList::pointerlist_t pointerlist)
        :images(pointerlist)
        {
            images.erase(
                std::remove_if(images.begin(), images.end(),
                            [](ImageList::pointer_t p) { return p == nullptr; }), images.end());
        }
    
    ImageList::ImageList(ImageList::vector_t&& vector)
        :images(std::move(vector))
        {
            images.erase(
                std::remove_if(images.begin(), images.end(),
                            [](ImageList::pointer_t p) { return p == nullptr; }), images.end());
        }
    
    ImageList::ImageList(ImageList&& other) noexcept
        :images(std::move(other.release()))
        {}
    
    ImageList& ImageList::operator=(ImageList&& other) noexcept {
        if (images != other.images) {
            images = std::move(other.release());
        }
        return *this;
    }
    
    ImageList::~ImageList() { reset(); }
    
    ImageList::vector_size_t ImageList::size() const          { return images.size(); }
    ImageList::iterator ImageList::begin()                    { return images.begin(); }
    ImageList::iterator ImageList::end()                      { return images.end(); }
    ImageList::const_iterator ImageList::begin() const        { return images.begin(); }
    ImageList::const_iterator ImageList::end() const          { return images.end(); }
    
    void ImageList::erase(ImageList::iterator it)             { images.erase(it); }
    void ImageList::prepend(ImageList::pointer_t image)       { images.insert(images.begin(), image); }
    void ImageList::push_front(ImageList::pointer_t image)    { images.insert(images.begin(), image); }
    void ImageList::append(ImageList::pointer_t image)        { images.push_back(image); }
    void ImageList::push_back(ImageList::pointer_t image)     { images.push_back(image); }
    void ImageList::push_back(ImageList::unique_t unique)     { images.push_back(unique.release()); }
    
    void ImageList::compute_sizes() const {
        computed_width = compute_width();
        computed_height = compute_height();
        computed_planes = compute_planes();
    }
    
    int ImageList::width() const  { return computed_width;  }
    int ImageList::height() const { return computed_height; }
    int ImageList::planes() const { return computed_planes; }
    
    ImageList::pointer_t ImageList::get(ImageList::vector_size_t idx) const   { return images[idx]; }
    ImageList::pointer_t ImageList::at(ImageList::vector_size_t idx) const    { return images.at(idx); }
    
    ImageList::unique_t ImageList::yank(ImageList::vector_size_t idx) {
        /// remove the pointer at idx, resizing the internal vector;
        /// return it as managed by a new unique_ptr
        /// ... this'll throw std::out_of_range if idx > images.size()
        ImageList::pointer_t outptr = images[idx];
        images.erase(
            std::remove_if(images.begin(),
                           images.end(),
                  [outptr](ImageList::pointer_t p) {
            return p == outptr ||
                   p == nullptr; }), images.end());
        images.shrink_to_fit();
        return unique_t(outptr);
    }
    
    ImageList::unique_t ImageList::pop() {
        ImageList::pointer_t outptr = images.back();
        images.pop_back();
        return unique_t(outptr);
    }
    
    void ImageList::reset() {
        ImageList::vector_size_t idx = 0,
                                 max = images.size();
        for (; idx != max; ++idx) { delete images[idx]; }
    }
    
    void ImageList::reset(ImageList::vector_t&& vector) {
        reset();
        images = std::move(vector);
    }
    
    ImageList::vector_t ImageList::release() {
        ImageList::vector_t out;
        out.swap(images);
        return out;
    }
    
    ImageList::vector_t ImageList::release(ImageList::vector_t&& vector) {
        ImageList::vector_t out = std::move(vector);
        out.swap(images);
        return out;
    }
    
    void ImageList::swap(ImageList& other) noexcept {
        images.swap(other.images);
    }
    
    std::size_t ImageList::hash(std::size_t seed) const noexcept {
        return std::accumulate(images.begin(),
                               images.end(),
                               seed, detail::rehasher_t());
    }
    
    DEFINE_DIMENSION_COMPUTE_FN(width);
    DEFINE_DIMENSION_COMPUTE_FN(height);
    DEFINE_DIMENSION_COMPUTE_FN(planes);
    
}

namespace std {
    
    template <>
    void swap(im::ImageList& p0, im::ImageList& p1) noexcept {
        p0.swap(p1);
    }
    
}; /* namespace std */
