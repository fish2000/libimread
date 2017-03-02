/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/store.hh>

namespace store {
    
    stringmap::stringmap_t stringmap::mapping() const {
        return stringmap::stringmap_t(cache);
    }
    
    bool stringmap::empty() const { return count() == 0; }
    std::size_t stringmap::size() const { return count(); }
    std::size_t stringmap::max_size() const noexcept { return stringvec_t().max_size(); }
    
    void stringmap::clear() {
        for (std::string const& key : list()) {
            del(key);
        }
    }
    
    bool stringmap::insert(std::pair<const std::string, std::string>&& item) {
        del(item.second);
        return set(item.first, item.second);
    }
    
    std::size_t stringmap::erase(std::string const& key) {
        return del(key);
    }
    
    std::string& stringmap::at(std::string const& key) {
        return get(key);
    }
    
    std::string const& stringmap::at(std::string const& key) const {
        return get(key);
    }
    
    std::string& stringmap::operator[](std::string const& key) {
        return get(key);
    }
    
    std::string& stringmap::operator[](std::string&& key) {
        return get(key);
    }
    
    std::size_t stringmap::count(std::string const& key) const {
        return static_cast<std::size_t>(get(key) == stringmap::base_t::null_value());
    }
    
} /// namespace store
