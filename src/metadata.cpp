/// Copyright 2012-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/metadata.hh>

namespace im {
    
    ImageWithMetadata::ImageWithMetadata()
        :meta("")
        {}
    ImageWithMetadata::ImageWithMetadata(std::string const& m)
        :meta(m)
        {}
    
    ImageWithMetadata::~ImageWithMetadata() {}
    
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
    
    
} /* namespace im */