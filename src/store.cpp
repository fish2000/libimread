/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/store.hh>
#include <libimread/serialization.hh>
#include <libimread/rehash.hh>

#define STRINGNULL() stringmapper::base_t::null_value()

namespace store {
    
    #pragma mark -
    #pragma mark base class store::stringmapper default methods
    
    void stringmapper::with_ini(std::string const& inistr) {
        detail::ini_impl(inistr, this);
    }
    
    void stringmapper::with_json(std::string const& jsonstr) {
        detail::json_impl(jsonstr, this);
    }
    
    void stringmapper::with_plist(std::string const& xmlstr) {
        detail::plist_impl(xmlstr, this);
    }
    
    void stringmapper::with_urlparam(std::string const& urlstr) {
        detail::urlparam_impl(urlstr, this);
    }
    
    void stringmapper::with_yaml(std::string const& yamlstr) {
        detail::yaml_impl(yamlstr, this);
    }
    
    void stringmapper::warm_cache() const {
        if (cache.size() < count()) {
            /// call get() for each key to warm the cache
            for (std::string const& key : list()) { get(key); }
        }
    }
    
    stringmapper::stringmap_t& stringmapper::mapping() const {
        warm_cache();
        return cache;
    }
    
    std::string stringmapper::mapping_ini() const {
        warm_cache();
        return detail::ini_dumps(cache);
    }
    
    std::string stringmapper::mapping_json() const {
        warm_cache();
        return detail::json_dumps(cache);
    }
    
    std::string stringmapper::mapping_plist() const {
        warm_cache();
        return detail::plist_dumps(cache);
    }
    
    std::string stringmapper::mapping_urlparam(bool questionmark) const {
        warm_cache();
        return detail::urlparam_dumps(cache, questionmark);
    }
    
    std::string stringmapper::mapping_yaml() const {
        warm_cache();
        return detail::yaml_dumps(cache);
    }
    
    std::string stringmapper::to_string() const {
        warm_cache();
        return detail::json_dumps(cache);
    }
    
    bool stringmapper::dump(std::string const& destination, bool overwrite, formatter format) const {
        warm_cache();
        switch (format) {
            case formatter::ini:
                return detail::dump(detail::ini_dumps(cache),
                                    destination,
                                    overwrite);
            case formatter::plist:
                return detail::dump(detail::plist_dumps(cache),
                                    destination,
                                    overwrite);
            case formatter::yaml:
                return detail::dump(detail::yaml_dumps(cache),
                                    destination,
                                    overwrite);
            case formatter::urlparam:
                return detail::dump(detail::urlparam_dumps(cache),
                                    destination,
                                    overwrite);
            case formatter::undefined:
            case formatter::json:
            default:
                return detail::dump(detail::json_dumps(cache),
                                    destination,
                                    overwrite);
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
        hash::rehash<std::string>(H, detail::json_dumps(cache));
        return H;
    }
    
    void stringmapper::clear() {
        for (std::string const& key : list()) { del(key); }
    }
    
    bool stringmapper::insert(std::pair<const std::string, std::string>&& item) {
        bool out = del(item.first);
        out &= set(item.first, item.second);
        return out;
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
    
    std::string& xattrmap::get_force(std::string const& key) {
        std::string val(xattr(key));
        if (val != STRINGNULL()) {
            cache[key] = val;
            return cache[key];
        }
        return STRINGNULL();
    }
    
    std::string const& xattrmap::get_force(std::string const& key) const {
        std::string val(xattr(key));
        if (val != STRINGNULL()) {
            cache[key] = val;
            return cache[key];
        }
        return STRINGNULL();
    }
    
    std::string& xattrmap::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        }
        return get_force(key);
    }
    
    std::string const& xattrmap::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        }
        return get_force(key);
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
            case stringmapper::formatter::ini:
                with_ini(serialized);
                break;
            case stringmapper::formatter::plist:
                with_plist(serialized);
                break;
            case stringmapper::formatter::yaml:
                with_yaml(serialized);
                break;
            case stringmapper::formatter::urlparam:
                with_urlparam(serialized);
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
        std::string serialized("");
        
        try {
            serialized = detail::load(source);
        } catch (im::FileSystemError&) {
            return out;
        } catch (im::CannotReadError&) {
            return out;
        }
        
        switch (detail::for_path(source)) {
            case stringmapper::formatter::ini: {
                detail::ini_impl(serialized, &out);
                return out;
            }
            case stringmapper::formatter::plist: {
                detail::plist_impl(serialized, &out);
                return out;
            }
            case stringmapper::formatter::yaml: {
                detail::yaml_impl(serialized, &out);
                return out;
            }
            case stringmapper::formatter::urlparam: {
                detail::urlparam_impl(serialized, &out);
                return out;
            }
            case stringmapper::formatter::undefined:
            case stringmapper::formatter::json:
            default: {
                detail::json_impl(serialized, &out);
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
#pragma mark std::hash<…> specializations for store::stringmap

namespace std {
    
    using store_hasher_t = std::hash<store::stringmapper>;
    using store_arg_t = store_hasher_t::argument_type;
    using store_out_t = store_hasher_t::result_type;
    
    store_out_t store_hasher_t::operator()(store_arg_t const& strmapper) const {
        return static_cast<store_out_t>(strmapper.hash());
    }
    
} /// namespace std