/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <plist/plist++.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/JSON/json11.h>
#include <libimread/store.hh>

#define STRINGNULL() stringmapper::base_t::null_value()

namespace store {
    
    #pragma mark -
    #pragma mark serialization helper implementations
    
    namespace detail {
        
        static void json_map_impl(Json const& jsonmap, stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr)       { return; }     /// `stringmap_ptr` must be a valid pointer
            if (jsonmap.type() != Type::OBJECT) { return; }     /// `jsonmap` must be a JSON map (née “Object”)
            for (std::string const& key : jsonmap.keys()) {
                stringmap_ptr->set(key, jsonmap.get(key));
            }
        }
        
        static void json_impl(Json const& json, stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            switch (json.type()) {
                case Type::OBJECT:
                    json_map_impl(json, stringmap_ptr);
                    return;
                case Type::ARRAY: {
                    std::size_t max = json.size();
                    if (max > 0) {
                        for (std::size_t idx = 0; idx < max; ++idx) {
                            json_map_impl(json[idx], stringmap_ptr);
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
        
        static void plist_dump(PList::Dictionary const& dict, std::string const& dest, bool overwrite = false) {
            using filesystem::path;
            using filesystem::NamedTemporaryFile;
            std::string destination(dest);
            
            try {
                destination = path::expand_user(dest).make_absolute().str();
            } catch (im::FileSystemError& exc) {
                throw;
            }
            
            if (path::exists(destination)) {
                if (!overwrite) {
                    imread_raise(PListIOError,
                        "store::detail::plist_dump(destination, overwrite=false): existant destination",
                     FF("\tdest        == %s", dest.c_str()),
                     FF("\tdestination == %s", destination.c_str()),
                        "\t(Requires overwrite=true or a unique destination)");
                } else if (path::is_directory(destination)) {
                    imread_raise(PListIOError,
                        "store::detail::plist_dump(destination): directory as existant destination",
                     FF("\toverwrite   == %s", overwrite ? "true" : "false"),
                     FF("\tdest        == %s", dest.c_str()),
                     FF("\tdestination == %s", destination.c_str()),
                        "\t(Requires overwrite=true with a non-directory destination)");
                }
            }
            
            NamedTemporaryFile tf(".plist");
            tf.open();
            tf.stream << dict.ToXml();
            tf.close();
            
            if (path::exists(destination)) {
                path::remove(destination);
            }
            
            path finalfile = tf.filepath.duplicate(destination);
            if (!finalfile.is_file()) {
                imread_raise(PListIOError,
                    "store::detail::plist_dump(destination, ...): failed writing to destination",
                 FF("\toverwrite   == %s", overwrite ? "true" : "false"),
                 FF("\tdest        == %s", dest.c_str()),
                 FF("\tdestination == %s", destination.c_str()),
                 FF("\tfinalfile   == %s", finalfile.c_str()));
            }
        }
        
        static PList::Dictionary plist_load(std::string const& source) {
            using filesystem::path;
            
            if (!path::exists(source)) {
                imread_raise(PListIOError,
                    "store::detail::plist_load(source): nonexistant source file",
                 FF("\tsource == %s", source.c_str()));
            }
            if (!path::is_file_or_link(source)) {
                imread_raise(PListIOError,
                    "store::detail::plist_load(source): non-file-or-link source file",
                 FF("\tsource == %s", source.c_str()));
            }
            if (!path::is_readable(source)) {
                imread_raise(PListIOError,
                    "store::detail::plist_load(source): unreadable source file",
                 FF("\tsource == %s", source.c_str()));
            }
            
            std::fstream stream;
            stream.open(source, std::ios::in);
            if (!stream.is_open()) {
                imread_raise(PListIOError,
                    "store::detail::plist_load(source): couldn't open a stream to read source file",
                 FF("\tsource == %s", source.c_str()));
            }
            
            /// Adapted from https://stackoverflow.com/a/3203502/298171
            std::string xml(std::istreambuf_iterator<char>(stream), {});
            stream.close();
            
            /// return by value
            return PList::Dictionary::FromXml(xml);
        }
        
        static void plist_impl(PList::Dictionary& dict, stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            std::for_each(dict.Begin(), dict.End(),
                      [&](auto const& kv) {
                            stringmap_ptr->set(
                                kv.first,
                                static_cast<PList::String*>(kv.second)->GetValue());
            });
        }
        
    }
    
    #pragma mark -
    #pragma mark base class store::stringmapper default methods
    
    void stringmapper::with_json(std::string const& jsonstr) {
        Json json = Json::parse(jsonstr);
        detail::json_impl(json, this);
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
    
    std::string stringmapper::to_string() const {
        warm_cache();
        return Json(cache).format();
    }
    
    bool stringmapper::dump(std::string const& destination, bool overwrite, formatter format) const {
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
    
    std::string& stringmapper::operator[](std::string const& key) {
        return get(key);
    }
    
    std::string& stringmapper::operator[](std::string&& key) {
        return get(key);
    }
    
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
    
    stringmap::stringmap(std::string const& jsonstr) {
        with_json(jsonstr);
    }
    
    stringmap stringmap::load_map(std::string const& source) {
        /// load_map() is a static function, there is no `this`:
        Json loadee = Json::null;
        stringmap out;
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
    
    void stringmap::warm_cache() const {
        /// NO-OP: the cache is the only backend --
        /// warming it just wastes a bunch of ops
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
