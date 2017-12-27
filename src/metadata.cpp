/// Copyright 2012-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <utility>
#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/metadata.hh>

/// Shortcut to std::string{ NULL_STR } value
#define STRINGNULL() store::detail::value_for_null<std::string>()

namespace im {
    
    Metadata::Metadata() {
        values.set("meta", ""); /// legacy, ugh
    }
    
    Metadata::Metadata(std::string const& m) {
        values.set("meta", m);
    }
    
    Metadata::~Metadata() {}
    
    /// member swap
    void Metadata::swap(Metadata& other) noexcept {
        using std::swap;
        swap(values,  other.values);
    }
    
    /// friend swap
    void swap(Metadata& lhs, Metadata& rhs) noexcept {
        lhs.swap(rhs);
    }
    
    bool operator==(Metadata const& lhs, Metadata const& rhs) {
        return ::operator==(lhs.values, rhs.values);
    }
    
    bool operator!=(Metadata const& lhs, Metadata const& rhs) {
        return ::operator!=(lhs.values, rhs.values);
    }
    
    Metadata::Metadata(Metadata const& other)
        :values(other.values)
        {}
    
    Metadata::Metadata(Metadata&& other) noexcept
        :values(std::move(other.values))
        {}
    
    Metadata& Metadata::operator=(Metadata const& other) {
        if (std::addressof(other.values) != std::addressof(values)) {
            values.clear();
            store::value_copy(other.values, values);
        }
        return *this;
    }
    
    Metadata& Metadata::operator=(Metadata&& other) noexcept {
        if (std::addressof(other.values) != std::addressof(values)) {
            values = std::exchange(other.values, values);
        }
        return *this;
    }
    
    std::string&       Metadata::get(std::string const& key)               { return values.get(key);           }
    std::string const& Metadata::get(std::string const& key) const         { return values.get(key);           }
    bool Metadata::set(std::string const& key, std::string const& value)   { return values.set(key, value);    }
    bool Metadata::del(std::string const& key)                             { return values.del(key);           }
    std::size_t Metadata::count() const                                    { return values.count();            }
    store::stringmap::stringvec_t Metadata::list() const                   { return values.list();             }
    
    bool Metadata::has_meta() const { return values.get("meta") != STRINGNULL(); }
    std::string const& Metadata::get_meta() const { return values.get("meta"); }
    std::string const& Metadata::set_meta(std::string const& m) { values.set("meta", m);
                                                           return values.get("meta"); }
    
    bool Metadata::has_icc_name() const { return values.get("icc_name") != STRINGNULL(); }
    std::string const& Metadata::get_icc_name() const { return values.get("icc_name"); }
    std::string const& Metadata::set_icc_name(std::string const& nm) { values.set("icc_name", nm);
                                                                return values.get("icc_name"); }
    
    bool Metadata::has_icc_data() const { return values.get("icc_data") != STRINGNULL(); }
    
    bytevec_t Metadata::get_icc_data() const {
        std::string const& datum = values.get("icc_data");
        bytevec_t out;
        out.resize(datum.size());
        std::copy(datum.begin(),
                  datum.end(),
                  out.begin());
        return out;
    }
    
    std::string const& Metadata::set_icc_data(std::string const& icc) {
        values.set("icc_data", icc);
        return values.get("icc_data");
    }
    
    std::string const& Metadata::set_icc_data(bytevec_t const& icc) {
        std::string datum;
        datum.resize(icc.size());
        std::copy(icc.begin(),
                  icc.end(),
                  datum.begin());
        return set_icc_data(datum);
    }
    
    std::string const& Metadata::set_icc_data(byte* data, std::size_t len) {
        std::string datum(data, data + len);
        return set_icc_data(datum);
    }
    
    std::size_t Metadata::hash(std::size_t H) const {
        return values.hash(H);
    }
    
} /* namespace im */

namespace std {
    
    #pragma mark -
    #pragma mark std::swap<…>() specializations for im::Metadata
    
    template <>
    void swap(im::Metadata& lhs, im::Metadata& rhs) noexcept {
        lhs.swap(rhs);
    }
    
    #pragma mark -
    #pragma mark std::hash<…> specializations for im::Metadata
    
    using metadata_hasher_t = std::hash<im::Metadata>;
    using metadata_arg_t = metadata_hasher_t::argument_type;
    using metadata_out_t = metadata_hasher_t::result_type;
    
    metadata_out_t metadata_hasher_t::operator()(metadata_arg_t const& md) const {
        return static_cast<metadata_out_t>(md.hash());
    }
    
} /* namespace std */