/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_METADATA_HH_
#define LIBIMREAD_METADATA_HH_

#include <vector>
#include <memory>
#include <string>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/store.hh>

namespace im {
    
    class Metadata {
        
        public:
            Metadata();
            Metadata(std::string const& m);
            virtual ~Metadata();
            
            Metadata(Metadata const&);
            Metadata(Metadata&&) noexcept;
            Metadata& operator=(Metadata const&);
            Metadata& operator=(Metadata&&) noexcept;
            
            inline std::string&       get(std::string const& key) { return values.get(key); };
            inline std::string const& get(std::string const& key) const { return values.get(key); };
            inline bool set(std::string const& key, std::string const& value) { return values.set(key, value); };
            inline bool del(std::string const& key) { return values.del(key); };
            inline std::size_t count() const { return values.count(); };
            inline store::stringmap::stringvec_t list() const { return values.list(); };
            
            bool has_meta() const;
            std::string const& get_meta() const;
            std::string const& set_meta(std::string const&);
            
            bool has_icc_name() const;
            std::string const& get_icc_name() const;
            std::string const& set_icc_name(std::string const&);
            
            bool has_icc_data() const;
            bytevec_t get_icc_data() const;
            std::string const& set_icc_data(std::string const&);
            std::string const& set_icc_data(bytevec_t const&);
            std::string const& set_icc_data(byte*, std::size_t);
            
        public:
            store::stringmap values;
    };
    
} /* namespace im */

#endif /// LIBIMREAD_METADATA_HH_