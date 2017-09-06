/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/pystring.hh>
#include <libimread/store.hh>
#include <libimread/serialization.hh>
#include <libimread/rehash.hh>

#define STRINGNULL() stringmapper::base_t::null_value()

namespace store {
    
    #pragma mark -
    #pragma mark base class store::stringmapper default methods
    
    stringmapper::formatter stringmapper::for_path(std::string const& pth) {
        using filesystem::path;
        std::string ext = pystring::lower(path::extension(pth));
        if (ext == "json") {
            return stringmapper::formatter::json;
        } else if (ext == "plist") {
            return stringmapper::formatter::plist;
        } else if (ext == "pickle") {
            return stringmapper::formatter::pickle;
        } else if (ext == "ini") {
            return stringmapper::formatter::ini;
        }
        return stringmapper::default_format; /// JSON
    }
    
    void stringmapper::with_json(std::string const& jsonstr) {
        Json json = Json::parse(jsonstr);
        detail::json_impl(json, this);
    }
    
    void stringmapper::with_plist(std::string const& xmlstr) {
        PList::Dictionary dict = PList::Dictionary::FromXml(xmlstr);
        detail::plist_impl(dict, this);
    }
    
    void stringmapper::warm_cache() const {
        stringvec_t keys(list());
        if (cache.size() < keys.size()) {
            /// call get() for each key to warm the cache
            for (std::string const& key : keys) { get(key); }
        }
    }
    
    stringmapper::stringmap_t& stringmapper::mapping() const {
        warm_cache();
        return cache;
    }
    
    std::string stringmapper::mapping_json() const {
        warm_cache();
        return Json(cache).format();
    }
    
    std::string stringmapper::mapping_plist() const {
        warm_cache();
        PList::Dictionary dict;
        for (auto const& item : cache) {
            dict.Set(item.first, PList::String(item.second));
        }
        return dict.ToXml();
    }
    
    std::string stringmapper::to_string() const {
        warm_cache();
        return Json(cache).format();
    }
    
    bool stringmapper::dump(std::string const& destination, bool overwrite, formatter format) const {
        /// only JSON works for now:
        warm_cache();
        switch (format) {
            case formatter::plist:
                return detail::plist_dump(cache, destination, overwrite);
            case formatter::undefined:
            case formatter::json:
            default:
                return detail::json_dump(cache, destination, overwrite);
        }
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
    
    std::size_t stringmapper::hash(std::size_t H) const {
        warm_cache();
        hash::rehash<std::string>(H, Json(cache).format());
        return H;
    }
    
    void stringmapper::clear() {
        for (std::string const& key : list()) { del(key); }
    }
    
    bool stringmapper::insert(std::pair<const std::string, std::string>&& item) {
        return (del(item.first) && set(item.first, item.second));
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
    
    // std::string& stringmapper::operator[](std::string const& key) {
    //     return get(key);
    // }
    
    // std::string& stringmapper::operator[](std::string&& key) {
    //     return get(key);
    // }
    
    std::size_t stringmapper::count(std::string const& key) const {
        return static_cast<std::size_t>(get(key) == STRINGNULL());
    }
    
    #pragma mark -
    #pragma mark store::xattrmap methods
    
    bool xattrmap::can_store() const noexcept { return true; }
    
    std::string& xattrmap::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        }
        std::string val(xattr(key));
        if (val != STRINGNULL()) {
            cache[key] = val;
            return cache[key];
        }
        return STRINGNULL();
    }
    
    std::string const& xattrmap::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        }
        std::string val(xattr(key));
        if (val != STRINGNULL()) {
            cache[key] = val;
            return cache[key];
        }
        return STRINGNULL();
    }
    
    bool xattrmap::set(std::string const& key, std::string const& value) {
        if (value == STRINGNULL()) { return del(key); }
        cache[key] = value;
        return xattr(key, value) == value;
    }
    
    bool xattrmap::del(std::string const& key) {
        if (cache.find(key) != cache.end()) { cache.erase(key); }
        return xattr(key, STRINGNULL()) == STRINGNULL();
    }
    
    std::size_t xattrmap::count() const {
        return xattrcount();
    }
    
    stringmapper::stringvec_t xattrmap::list() const {
        return xattrs();
    }
    
    #pragma mark -
    #pragma mark store::stringmap methods
    
    bool stringmap::can_store() const noexcept { return true; }
    
    stringmap::stringmap() noexcept {}
    
    stringmap::stringmap(std::string const& serialized,
                         stringmapper::formatter format) {
        switch (format) {
            case stringmapper::formatter::plist:
                with_plist(serialized);
                break;
            case stringmapper::formatter::undefined:
            case stringmapper::formatter::json:
            default:
                with_json(serialized);
                break;
        }
    }
    
    stringmap stringmap::load_map(std::string const& source) {
        /// load_map() is a static function, there is no `this`:
        stringmap out;
        switch (stringmapper::for_path(source)) {
            case stringmapper::formatter::plist: {
                PList::Dictionary dict;
                try {
                    /// PList::Dictionary::operator=(…) isn’t overloaded
                    /// for rvalue refs, hence this idiotic value do-si-do:
                    auto d = detail::plist_load(source);
                    dict = d;
                } catch (im::FileSystemError&) {
                    return out;
                } catch (im::PListIOError&) {
                    return out;
                }
                detail::plist_impl(dict, &out);
                return out;
            }
            case stringmapper::formatter::undefined:
            case stringmapper::formatter::json:
            default: {
                Json loadee = Json::null;
                try {
                    loadee = Json::load(source);
                } catch (im::FileSystemError&) {
                    return out;
                } catch (im::JSONIOError&) {
                    return out;
                }
                detail::json_impl(loadee, &out);
                return out;
            }
        }
    }
    
    void stringmap::warm_cache() const {
        /// NO-OP: the cache is the only backend --
        /// warming it just wastes a bunch of ops
    }
    
    void stringmap::swap(stringmap& other) noexcept {
        using std::swap;
        swap(cache, other.cache);
    }
    
    std::string& stringmap::get(std::string const& key) {
        if (cache.find(key) != cache.end()) { return cache[key]; }
        return STRINGNULL();
    }
    
    std::string const& stringmap::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) { return cache[key]; }
        return STRINGNULL();
    }
    
    bool stringmap::set(std::string const& key, std::string const& value) {
        cache[key] = value;
        return cache.count(key) == 1;
    }
    
    bool stringmap::del(std::string const& key) {
        if (cache.find(key) != cache.end()) { cache.erase(key); }
        return cache.count(key) == 0;
    }
    
    std::size_t stringmap::count() const {
        return cache.size();
    }
    
    stringmapper::stringvec_t stringmap::list() const {
        stringmapper::stringvec_t out{};
        out.reserve(cache.size());
        for (auto const& item : cache) {
            out.emplace_back(item.first);
        }
        return out;
    }
    
} /// namespace store

#pragma mark -
#pragma mark std::swap<…>() specializations for store::stringmap

namespace std {
    
    using store_hasher_t = std::hash<store::stringmap>;
    using store_arg_t = store_hasher_t::argument_type;
    using store_out_t = store_hasher_t::result_type;
    
    store_out_t store_hasher_t::operator()(store_arg_t const& s) const {
        return static_cast<store_out_t>(s.hash());
    }
    
} /// namespace std