/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/JSON/json11.h>
#include <libimread/ext/pystring.hh>
#include <libimread/store.hh>

namespace store {
    
    namespace detail {
        
        inline void json_map_impl(Json const& jsonmap, stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr)       { return; }
            if (jsonmap.type() != Type::OBJECT) { return; }
            for (std::string const& key : jsonmap.keys()) {
                stringmap_ptr->set(key, jsonmap.get(key));
            }
        }
        
        inline void json_impl(Json const& jsonmap, stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }
            switch (jsonmap.type()) {
                case Type::OBJECT:
                    json_map_impl(jsonmap, stringmap_ptr);
                    return;
                case Type::ARRAY: {
                    std::size_t max = jsonmap.size();
                    if (max > 0) {
                        for (std::size_t idx = 0; idx < max; ++idx) {
                            json_map_impl(jsonmap[idx], stringmap_ptr);
                        }
                    }
                    return;
                }
                case Type::JSNULL:
                case Type::BOOLEAN:
                case Type::NUMBER:
                case Type::STRING:
                default:
                    return;
            }
        }
        
    }
    
    #pragma mark - base class store::stringmapper default methods
    
    void stringmapper::with_json(std::string const& jsonstr) {
        Json jsonmap = Json::parse(jsonstr);
        detail::json_impl(jsonmap, this);
    }
    
    void stringmapper::warm_cache() const {
        /// call get() for each key to warm the cache:
        for (std::string const& key : list()) { get(key); }
    }
    
    stringmapper::stringmap_t& stringmapper::mapping() const {
        warm_cache();
        return cache;
    }
    
    std::string stringmapper::mapping_json() const {
        warm_cache();
        return Json(cache).format();
    }
    
    bool stringmapper::dump(std::string const& destination, formatter format, bool overwrite) const {
        /// only JSON works for now:
        warm_cache();
        Json dumpee(cache);
        try {
            dumpee.dump(destination, overwrite);
        } catch (im::FileSystemError&) {
            return false;
        } catch (im::JSONIOError&) {
            return false;
        }
        return true;
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
    
    stringmap stringmap::load_map(std::string const& source) {
        /// load_map() is a static function, there is no `this`:
        Json loadee = Json::null;
        stringmap out;
        try {
            loadee = Json::load(source);
        } catch (im::FileSystemError& exc) {
            WTF("FileSystemError:", exc.what());
            return out;
        } catch (im::JSONIOError& exc) {
            WTF("JSONIOError:", exc.what());
            return out;
        }
        detail::json_impl(loadee, &out);
        return out;
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
