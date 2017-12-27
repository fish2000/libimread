/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_METADATA_HH_
#define LIBIMREAD_METADATA_HH_

#include <vector>
#include <string>
#include <libimread/libimread.hpp>
#include <libimread/store.hh>

namespace im {
    
    using stringvec_t = std::vector<std::string>;
    
    class Metadata : public store::stringmapper {
        
        public:
            Metadata();
            Metadata(std::string const&);
            virtual ~Metadata();
        
        public:
            DECLARE_STRINGMAPPER_TEMPLATES(Metadata);
        
        public:
            void swap(Metadata&) noexcept;
            friend void swap(Metadata&, Metadata&) noexcept;
            friend bool operator==(Metadata const&, Metadata const&);
            friend bool operator!=(Metadata const&, Metadata const&);
            
        public:
            Metadata(Metadata const&);
            Metadata(Metadata&&) noexcept;
            Metadata& operator=(Metadata const&);
            Metadata& operator=(Metadata&&) noexcept;
            
        public:
            virtual std::string&       get(std::string const&) override;
            virtual std::string const& get(std::string const&) const override;
            virtual bool set(std::string const&, std::string const&) override;
            virtual bool del(std::string const&) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
            
        public:
            bool has_meta() const;
            std::string const& get_meta() const;
            std::string const& set_meta(std::string const&);
            
        public:
            bool has_icc_name() const;
            std::string const& get_icc_name() const;
            std::string const& set_icc_name(std::string const&);
            
        public:
            bool has_icc_data() const;
            bytevec_t get_icc_data() const;
            std::string const& set_icc_data(std::string const&);
            std::string const& set_icc_data(bytevec_t const&);
            std::string const& set_icc_data(byte*, std::size_t);
            
        public:
            virtual std::size_t hash(std::size_t H = 0) const override;
            
        protected:
            mutable store::stringmap values;
    };
    
} /* namespace im */

namespace std {
    
    template <>
    void swap(im::Metadata&, im::Metadata&) noexcept;
    
    /// std::hash specialization for im::Metadata
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::Metadata> {
        
        typedef im::Metadata argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& meta) const;
        
    };
    
} /* namespace std */

#endif /// LIBIMREAD_METADATA_HH_