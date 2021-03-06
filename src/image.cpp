/// Copyright 2012-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/metadata.hh>
#include <libimread/rehash.hh>

namespace im {
    
    Image::~Image() {}
    
    void* Image::rowp() const {
        return rowp(0);
    }
    
    int Image::nbytes() const {
        const int bits = nbits();
        return (bits / 8) + bool(bits % 8);
    }
    
    int Image::min(int dimension) const {
        /// default implementation: DOES NOTHING
        return 0;
    }
    
    int Image::dim_or(int dimension, int default_value) const {
        if (dimension >= ndims()) { return default_value; }
        return dim(dimension);
    }
    
    int Image::stride_or(int dimension, int default_value) const {
        if (dimension >= ndims()) { return default_value; }
        return stride(dimension);
    }
    
    int Image::min_or(int dimension, int default_value) const {
        if (dimension >= ndims()) { return default_value; }
        return min(dimension);
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
    
    int Image::left() const {
        if (ndims() < 1) { return 0; }
        return min(0);
    }
    
    int Image::right() const {
        if (ndims() < 1) { return 0; }
        return min(0) + dim(0) - 1;
    }
    
    int Image::top() const {
        if (ndims() < 2) { return 0; }
        return min(1);
    }
    
    int Image::bottom() const {
        if (ndims() < 2) { return 0; }
        return min(1) + dim(1) - 1;
    }
    
    float Image::entropy() const {
        if (!histo.get()) {
            histo = std::make_shared<Histogram>(this);
        }
        return histo->entropy();
    }
    
    int Image::otsu() const {
        if (!histo.get()) {
            histo = std::make_shared<Histogram>(this);
        }
        return histo->otsu();
    }
    
    Metadata& Image::metadata() {
        return md;
    }
    
    Metadata const& Image::metadata() const {
        return md;
    }
    
    Metadata& Image::metadata(Metadata& new_md) {
        md = Metadata(new_md);
        return md;
    }
    
    Metadata& Image::metadata(Metadata&& new_md) {
        md = std::move(new_md);
        return md;
    }
    
    Metadata* Image::metadata_ptr() {
        return &md;
    }
    
    Metadata const* Image::metadata_ptr() const {
        return &md;
    }
    
    ImageFactory::~ImageFactory() {}
    
}
