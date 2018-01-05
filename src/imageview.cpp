/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/imageview.hh>
#include <libimread/histogram.hh>
#include <libimread/rehash.hh>

namespace im {
    
    ImageView::ImageView(ImageView const& other)
        :source(other.source)
        {}
        
    ImageView::ImageView(ImageView&& other) noexcept
        :source(std::exchange(other.source, nullptr))
        ,histo(std::exchange(other.histo, nullptr))
        {}
    
    ImageView::ImageView(Image* image)
        :source(image)
        {}
    
    ImageView::~ImageView() {}
    
    ImageView& ImageView::operator=(ImageView const& other) & {
        if (source != other.source) {
            ImageView(other).swap(*this);
        }
        return *this;
    }
    
    ImageView& ImageView::operator=(ImageView&& other) & noexcept {
        if (source != other.source) {
            source = std::exchange(other.source, nullptr);
            histo = std::exchange(other.histo, nullptr);
        }
        return *this;
    }
    
    ImageView& ImageView::operator=(Image* image_ptr) & {
        if (source != image_ptr) {
            source = image_ptr;
        }
        return *this;
    }
    
    void* ImageView::rowp(int r) const          { return source->rowp(r); }
    void* ImageView::rowp() const               { return source->rowp(0); }
    int ImageView::nbits() const                { return source->nbits(); }
    int ImageView::nbytes() const               { return source->nbytes(); }
    int ImageView::ndims() const                { return source->ndims(); }
    int ImageView::dim(int d) const             { return source->dim(d); }
    int ImageView::stride(int s) const          { return source->stride(s); }
    int ImageView::min(int s) const             { return source->min(s); }
    bool ImageView::is_signed() const           { return source->is_signed(); }
    bool ImageView::is_floating_point() const   { return source->is_floating_point(); }
    
    int ImageView::dim_or(int dim, int default_value) const {
        if (dim >= source->ndims()) { return default_value; }
        return source->dim(dim);
    }
    
    int ImageView::stride_or(int dim, int default_value) const {
        if (dim >= source->ndims()) { return default_value; }
        return source->stride(dim);
    }
    
    int ImageView::min_or(int dim, int default_value) const {
        if (dim >= source->ndims()) { return default_value; }
        return source->min(dim);
    }
    
    int ImageView::width() const                { return source->dim(0); }
    int ImageView::height() const               { return source->dim(1); }
    int ImageView::planes() const               { return source->dim_or(2); }
    
    int ImageView::size() const                 { return source->dim_or(0) *
                                                         source->dim_or(1) *
                                                         source->dim_or(2) * 
                                                         source->dim_or(3); }
    
    int ImageView::left() const {
        if (source->ndims() < 1) { return 0; }
        return source->min(0);
    }
    
    int ImageView::right() const {
        if (source->ndims() < 1) { return 0; }
        return source->min(0) + source->dim(0) - 1;
    }
    
    int ImageView::top() const {
        if (source->ndims() < 2) { return 0; }
        return source->min(1);
    }
    
    int ImageView::bottom() const {
        if (source->ndims() < 2) { return 0; }
        return source->min(1) + source->dim(1) - 1;
    }
    
    // Histogram ImageView::histogram() const {
    //     return Histogram(source);
    // }
    
    float ImageView::entropy() const {
        if (!histo.get()) {
            histo = std::make_unique<Histogram>(source);
        }
        return histo->entropy();
    }
    
    int ImageView::otsu() const {
        if (!histo.get()) {
            histo = std::make_unique<Histogram>(source);
        }
        return histo->otsu();
    }
    
    Metadata& ImageView::metadata() {
        return source->md;
    }
    
    Metadata const& ImageView::metadata() const {
        return source->md;
    }
    
    Metadata& ImageView::metadata(Metadata& new_md) {
        source->md = Metadata(new_md);
        return source->md;
    }
    
    Metadata& ImageView::metadata(Metadata&& new_md) {
        source->md = std::move(new_md);
        return source->md;
    }
    
    Metadata* ImageView::metadata_ptr() {
        return &source->md;
    }
    
    Metadata const* ImageView::metadata_ptr() const {
        return &source->md;
    }
    
    ImageView::shared_imageview_t ImageView::shared() {
        return std::const_pointer_cast<ImageView>(shared_from_this());
    }
    
    ImageView::weak_imageview_t ImageView::weak() {
        return weak_imageview_t(
               std::const_pointer_cast<ImageView>(shared_from_this()));
    }
    
    std::size_t ImageView::hash(std::size_t seed) const noexcept {
        hash::rehash<ImageView::image_ptr_t>(seed, source);
        return seed;
    }
    
    void ImageView::swap(ImageView& other) {
        using std::swap;
        swap(source, other.source);
        swap(histo,  other.histo);
    }
    
    void swap(ImageView& lhs, ImageView& rhs) {
        using std::swap;
        swap(lhs.source, rhs.source);
        swap(lhs.histo,  rhs.histo);
    }
    
} /* namespace im */
