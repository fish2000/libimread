/// Copyright 2012-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
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
    
    bool operator==(Metadata const& lhs, Metadata const& rhs) {
        return &lhs == &rhs;
    }
    
    bool operator!=(Metadata const& lhs, Metadata const& rhs) {
        return &lhs != &rhs;
    }
    
    Metadata::Metadata(Metadata const& other)
        :values()
        { store::value_copy(other.values, values); }
    
    Metadata::Metadata(Metadata&& other) noexcept
        :values(std::move(other.values))
        {}
    
    Metadata& Metadata::operator=(Metadata const& other) {
        if (other != *this) {
            values = store::stringmap{};
            store::value_copy(other.values, values);
        }
        return *this;
    }
    
    Metadata& Metadata::operator=(Metadata&& other) noexcept {
        if (other != *this) {
            values = std::exchange(other.values, store::stringmap{});
        }
        return *this;
    }
    
    bool Metadata::has_meta() const { return values.get("meta") != STRINGNULL(); }
    std::string const& Metadata::get_meta() const { return values.get("meta"); }
    std::string const& Metadata::set_meta(std::string const& m) { values.set("meta", m); return values.get("meta"); }
    
    bool Metadata::has_icc_name() const { return values.get("icc_name") != STRINGNULL(); }
    std::string const& Metadata::get_icc_name() const { return values.get("icc_name"); }
    std::string const& Metadata::set_icc_name(std::string const& nm) { values.set("icc_name", nm); return values.get("icc_name"); }
    
    bool Metadata::has_icc_data() const { return values.get("icc_data") != STRINGNULL(); }
    
    bytevec_t Metadata::get_icc_data() const {
        std::string const& datum = values.get("icc_data");
        bytevec_t out;
        out.reserve(datum.size());
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
    
} /* namespace im */