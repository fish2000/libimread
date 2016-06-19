/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/rehash.hh>

namespace im {
    
    namespace detail {
        using rehasher_t = hash::rehasher<ImageList::pointer_t>;
    }
    
    int Image::nbytes() const {
        const int bits = this->nbits();
        return (bits / 8) + bool(bits % 8);
    }
    
    int Image::dim_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->dim(dim);
    }
    
    int Image::stride_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->stride(dim);
    }
    
    int Image::width() const {
        return dim(0);
    }
    
    int Image::height() const {
        return dim(1);
    }
    
    int Image::planes() const {
        return dim_or(2);
    }
    
    int Image::size() const {
        return dim_or(0) * dim_or(1) * dim_or(2) * dim_or(3);
    }
    
    Image::shared_image_t Image::shared() {
        return shared_from_this();
    }
    
    Image::const_shared_image_t Image::shared() const {
        return shared_from_this();
    }
    
    Image::weak_image_t Image::weak() {
        return weak_image_t(shared_from_this());
    }
    
    Image::const_weak_image_t Image::weak() const {
        return const_weak_image_t(shared_from_this());
    }
    
    ImageFactory::~ImageFactory() {}
    
    ImageWithMetadata::ImageWithMetadata()
        :meta("")
        {}
    ImageWithMetadata::ImageWithMetadata(std::string const& m)
        :meta(m)
        {}
    
    ImageWithMetadata::~ImageWithMetadata() {}
    
    std::string const& ImageWithMetadata::get_meta() const { return meta; }
    
    void ImageWithMetadata::set_meta(std::string const& m) { meta = m; }
    
    ImageList::ImageList(ImageList::pointerlist_t pointerlist)
        :content(pointerlist)
        {
            content.erase(
                std::remove_if(content.begin(), content.end(),
                            [](ImageList::pointer_t p) { return p == nullptr; }),
                content.end());
        }
    
    ImageList::ImageList(ImageList::vector_t&& vector)
        :content(std::move(vector))
        {
            content.erase(
                std::remove_if(content.begin(), content.end(),
                            [](ImageList::pointer_t p) { return p == nullptr; }),
                content.end());
        }
    
    ImageList::ImageList(ImageList&& other) noexcept
        :content(std::move(other.release()))
        {}
    
    ImageList& ImageList::operator=(ImageList&& other) noexcept {
        if (content != other.content) {
            content = std::move(other.release());
        }
        return *this;
    }
    
    ImageList::vector_size_t ImageList::size() const          { return content.size(); }
    ImageList::iterator ImageList::begin()                    { return content.begin(); }
    ImageList::iterator ImageList::end()                      { return content.end(); }
    ImageList::const_iterator ImageList::begin() const        { return content.begin(); }
    ImageList::const_iterator ImageList::end() const          { return content.end(); }
    
    void ImageList::erase(ImageList::iterator it)             { content.erase(it); }
    void ImageList::prepend(ImageList::pointer_t image)       { content.insert(content.begin(), image); }
    void ImageList::push_front(ImageList::pointer_t image)    { content.insert(content.begin(), image); }
    void ImageList::append(ImageList::pointer_t image)        { content.push_back(image); }
    void ImageList::push_back(ImageList::pointer_t image)     { content.push_back(image); }
    void ImageList::push_back(ImageList::unique_t unique)     { content.push_back(unique.release()); }
    
    ImageList::pointer_t ImageList::get(ImageList::vector_size_t idx) const   { return content[idx]; }
    ImageList::pointer_t ImageList::at(ImageList::vector_size_t idx) const    { return content.at(idx); }
    
    ImageList::unique_t ImageList::yank(ImageList::vector_size_t idx) {
        /// remove the pointer at idx, resizing the internal vector;
        /// return it as managed by a new unique_ptr
        /// ... this'll throw std::out_of_range if idx > content.size()
        ImageList::pointer_t outptr = content.at(idx);
        content.erase(
            std::remove_if(content.begin(), content.end(),
                  [outptr](ImageList::pointer_t p) { return p == outptr || p == nullptr; }),
            content.end());
        content.shrink_to_fit();
        return unique_t(outptr);
    }
    
    ImageList::unique_t ImageList::pop() {
        ImageList::pointer_t outptr = content.back();
        content.pop_back();
        return unique_t(outptr);
    }
    
    void ImageList::reset() {
        ImageList::vector_size_t idx = 0,
                                 max = content.size();
        for (; idx != max; ++idx) { delete content[idx]; }
    }
    void ImageList::reset(ImageList::vector_t&& vector) {
        reset();
        content = std::move(vector);
    }
    
    ImageList::~ImageList() { reset(); }
    
    ImageList::vector_t ImageList::release() {
        ImageList::vector_t out;
        out.swap(content);
        return out;
    }
    
    void ImageList::swap(ImageList& other) noexcept {
        content.swap(other.content);
    }
    
    std::size_t ImageList::hash(std::size_t seed) const noexcept {
        return std::accumulate(content.begin(), content.end(),
                               seed, detail::rehasher_t());
    }
    
}

namespace std {
    
    template <>
    void swap(im::ImageList& p0, im::ImageList& p1) noexcept {
        p0.swap(p1);
    }
    
}; /* namespace std */
