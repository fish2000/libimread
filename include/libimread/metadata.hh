/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_METADATA_HH_
#define LIBIMREAD_METADATA_HH_

#include <vector>
#include <memory>
#include <string>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    class ImageWithMetadata {
        
        public:
            ImageWithMetadata();
            ImageWithMetadata(std::string const& m);
            virtual ~ImageWithMetadata();
            
            bool has_meta() const;
            std::string const& get_meta() const;
            std::string const& set_meta(std::string const&);
            
            bool has_icc_name() const;
            std::string const& get_icc_name() const;
            std::string const& set_icc_name(std::string const&);
            
            bool has_icc_data() const;
            bytevec_t const& get_icc_data() const;
            bytevec_t const& set_icc_data(bytevec_t const&);
            bytevec_t const& set_icc_data(byte*, std::size_t);
            
        protected:
            std::string meta;
            std::string icc_name;
            bytevec_t icc_data;
    };
    
} /* namespace im */

#endif /// LIBIMREAD_METADATA_HH_