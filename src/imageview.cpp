/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/imageview.hh>
#include <libimread/image.hh>

namespace im {
    
    void* ImageView::rowp(int r) const {
        return source->rowp(r);
    }
    int ImageView::nbits() const {
        return source->nbits();
    }
    int ImageView::nbytes() const {
        return source->nbytes();
    }
    int ImageView::ndims() const {
        return source->ndims();
    }
    int ImageView::dim(int d) const {
        return source->dim(d);
    }
    int ImageView::stride(int s) const {
        return source->stride(s);
    }
    bool ImageView::is_signed() const {
        return source->is_signed();
    }
    bool ImageView::is_floating_point() const {
        return source->is_floating_point();
    }
    
    /// if we have successfully constructed `source`, we can
    /// go ahead and call source->shared() and source->weak()
    /// as the fuck much as we would like
    ///
    ///     auto s = source->shared(); /// fine to do in instance methods
    
    // int ImageView::nbytes() const {
    //     const int bits = this->nbits();
    //     return (bits / 8) + bool(bits % 8);
    // }
    
    int ImageView::dim_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->dim(dim);
    }
    
    int ImageView::stride_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->stride(dim);
    }
    
    int ImageView::width() const {
        return dim(0);
    }
    
    int ImageView::height() const {
        return dim(1);
    }
    
    int ImageView::planes() const {
        return dim_or(2);
    }
    
    int ImageView::size() const {
        return dim_or(0) * dim_or(1) * dim_or(2) * dim_or(3);
    }
    
    ImageView::shared_imageview_t ImageView::shared() {
        return std::const_pointer_cast<ImageView>(shared_from_this());
    }
    
    ImageView::weak_imageview_t ImageView::weak() {
        return weak_imageview_t(
               std::const_pointer_cast<ImageView>(shared_from_this()));
    }
    
} /* namespace im */
