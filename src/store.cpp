/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/JSON/json11.h>
#include <libimread/store.hh>

namespace store {
    
    #pragma mark - base class store::stringmapper default methods
    
    void stringmapper::with_json(std::string const& jsonstr) {
        Json jsonmap = Json::parse(jsonstr);
        if (jsonmap.type() == Type::OBJECT) {
            for (std::string const& key : jsonmap.keys()) {
                set(key, jsonmap.get(key));
            }
        } else if (jsonmap.type() == Type::ARRAY) {
            if (jsonmap.size() == 1) {
                Json jsonmap0 = jsonmap[0];
                if (jsonmap0.type() == Type::OBJECT) {
                    for (std::string const& key : jsonmap0.keys()) {
                        set(key, jsonmap0.get(key));
                    }
                }
            }
        }
    }
    
    void stringmapper::warm_cache() const {
        /// cache warmup: call get() for each key in the stringmapper
        for (std::string const& key : list()) { get(key); }
    }
    
    stringmapper::stringmap_t& stringmapper::mapping() const {
        warm_cache();
        // return stringmapper::stringmap_t(cache);
        return cache;
    }
    
    std::string stringmapper::mapping_json() const {
        warm_cache();
        return Json(cache).format();
    }
    
    stringmapper::~stringmapper() {}
    
    bool stringmapper::empty() const {
        return count() == 0;
    }
    
    std::size_t stringmapper::size() const {
        return count();
    }
    
    std::size_t stringmapper::max_size() const noexcept {
        return stringvec_t().max_size();
    }
    
    void stringmapper::clear() {
        for (std::string const& key : list()) {
            del(key);
        }
    }
    
    bool stringmapper::insert(std::pair<const std::string, std::string>&& item) {
        del(item.second);
        return set(item.first, item.second);
    }
    
    std::size_t stringmapper::erase(std::string const& key) {
        return del(key);
    }
    
    std::string& stringmapper::at(std::string const& key) {
        return get(key);
    }
    
    std::string const& stringmapper::at(std::string const& key) const {
        return get(key);
    }
    
    std::string& stringmapper::operator[](std::string const& key) {
        return get(key);
    }
    
    std::string& stringmapper::operator[](std::string&& key) {
        return get(key);
    }
    
    std::size_t stringmapper::count(std::string const& key) const {
        return static_cast<std::size_t>(get(key) == stringmapper::base_t::null_value());
    }
    
    #pragma mark - store::xattrmap methods
    
    bool xattrmap::can_store() const noexcept { return true; }
    
    std::string& xattrmap::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            std::string val(xattr(key));
            if (val != stringmapper::base_t::null_value()) {
                cache[key] = val;
                return cache[key];
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    std::string const& xattrmap::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            std::string val(xattr(key));
            if (val != stringmapper::base_t::null_value()) {
                cache[key] = val;
                return cache[key];
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    bool xattrmap::set(std::string const& key, std::string const& value) {
        cache[key] = value;
        return xattr(key, value) == value;
    }
    
    bool xattrmap::del(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            cache.erase(key);
        }
        return xattr(key, stringmapper::base_t::null_value()) == stringmapper::base_t::null_value();
    }
    
    std::size_t xattrmap::count() const {
        return xattrcount();
    }
    
    stringmapper::stringvec_t xattrmap::list() const {
        return xattrs();
    }
    
    #pragma mark - store::stringmap methods
    
    bool stringmap::can_store() const noexcept { return true; }
    
    stringmap::stringmap() noexcept {}
    
    stringmap::stringmap(std::string const& jsonstr) {
        with_json(jsonstr);
    }
    
    std::string& stringmap::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            return stringmapper::base_t::null_value();
        }
    }
    
    std::string const& stringmap::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            return stringmapper::base_t::null_value();
        }
    }
    
    bool stringmap::set(std::string const& key, std::string const& value) {
        cache[key] = value;
        return cache.at(key) == value;
    }
    
    bool stringmap::del(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            cache.erase(key);
        }
        return cache.count(key) == 0;
    }
    
    std::size_t stringmap::count() const {
        return cache.size();
    }
    
    stringmapper::stringvec_t stringmap::list() const {
        stringmapper::stringvec_t out{};
        out.reserve(count());
        for (auto const& item : cache) {
            out.emplace_back(item.first);
        }
        return out;
    }
    
} /// namespace store
