/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/histogram.hh>
#include <libimread/rehash.hh>

namespace im {
    
    void* Image::rowp() const {
        return this->rowp(0);
    }
    
    int Image::nbytes() const {
        const int bits = this->nbits();
        return (bits / 8) + bool(bits % 8);
    }
    
    int Image::min(int dim) const {
        return 0;
    }
    
    int Image::dim_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->dim(dim);
    }
    
    int Image::stride_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->stride(dim);
    }
    
    int Image::min_or(int dim, int default_value) const {
        if (dim >= this->ndims()) { return default_value; }
        return this->min(dim);
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
    
    Histogram Image::histogram() const {
        return Histogram(this);
    }
    
    float Image::entropy() const {
        Histogram histo(this);
        return histo.entropy();
    }
    
    int Image::otsu() const {
        Histogram histo(this);
        return histo.otsu();
    }
    
    ImageFactory::~ImageFactory() {}
    
    ImageWithMetadata::ImageWithMetadata()
        :meta("")
        {}
    ImageWithMetadata::ImageWithMetadata(std::string const& m)
        :meta(m)
        {}
    
    ImageWithMetadata::~ImageWithMetadata() {}
    
    using bytevec_t = ImageWithMetadata::bytevec_t;
    
    bool ImageWithMetadata::has_meta() const { return !meta.empty(); }
    std::string const& ImageWithMetadata::get_meta() const { return meta; }
    std::string const& ImageWithMetadata::set_meta(std::string const& m) { meta = m; return meta; }
    
    bool ImageWithMetadata::has_icc_name() const { return !icc_name.empty(); }
    std::string const& ImageWithMetadata::get_icc_name() const { return icc_name; }
    std::string const& ImageWithMetadata::set_icc_name(std::string const& nm) { icc_name = nm; return icc_name; }
    
    bool ImageWithMetadata::has_icc_data() const { return !icc_data.empty(); }
    bytevec_t const& ImageWithMetadata::get_icc_data() const { return icc_data; }
    bytevec_t const& ImageWithMetadata::set_icc_data(bytevec_t const& icc) { icc_data = icc; return icc_data; }
    bytevec_t const& ImageWithMetadata::set_icc_data(byte* data, std::size_t len) { icc_data = bytevec_t(data, data + len); return icc_data; }
}
