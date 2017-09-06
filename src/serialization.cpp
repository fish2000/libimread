/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/serialization.hh>

namespace store {
    
    #pragma mark -
    #pragma mark serialization helper implementations
    
    namespace detail {
        
        #pragma mark -
        #pragma mark JSON serialization helpers
        
        bool json_dump(store::stringmapper::stringmap_t const& cache, std::string const& destination, bool overwrite) {
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
        
        void json_map_impl(Json const& jsonmap, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr)       { return; }     /// `stringmap_ptr` must be a valid pointer
            if (jsonmap.type() != Type::OBJECT) { return; }     /// `jsonmap` must be a JSON map (née “Object”)
            for (std::string const& key : jsonmap.keys()) {
                stringmap_ptr->set(key, static_cast<std::string>(jsonmap.get(key)));
            }
        }
        
        void json_impl(Json const& json, store::stringmapper* stringmap_ptr) {
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
        
        #pragma mark -
        #pragma mark Property list (plist) serialization helpers
        
        bool plist_dump(store::stringmapper::stringmap_t const& cache, std::string const& dest, bool overwrite) {
            using filesystem::path;
            using filesystem::NamedTemporaryFile;
            std::string destination(dest);
            
            PList::Dictionary dict;
            for (auto const& item : cache) {
                dict.Set(item.first, PList::String(item.second));
            }
            
            try {
                destination = path::expand_user(dest).make_absolute().str();
            } catch (im::FileSystemError& exc) {
                // throw;
                return false;
            }
            
            if (path::exists(destination)) {
                if (!overwrite) {
                    // imread_raise(PListIOError,
                    //     "store::detail::plist_dump(destination, overwrite=false): existant destination",
                    //  FF("\tdest        == %s", dest.c_str()),
                    //  FF("\tdestination == %s", destination.c_str()),
                    //     "\t(Requires overwrite=true or a unique destination)");
                    return false;
                } else if (path::is_directory(destination)) {
                    // imread_raise(PListIOError,
                    //     "store::detail::plist_dump(destination): directory as existant destination",
                    //  FF("\toverwrite   == %s", overwrite ? "true" : "false"),
                    //  FF("\tdest        == %s", dest.c_str()),
                    //  FF("\tdestination == %s", destination.c_str()),
                    //     "\t(Requires overwrite=true with a non-directory destination)");
                    return false;
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
                // imread_raise(PListIOError,
                //     "store::detail::plist_dump(destination, ...): failed writing to destination",
                //  FF("\toverwrite   == %s", overwrite ? "true" : "false"),
                //  FF("\tdest        == %s", dest.c_str()),
                //  FF("\tdestination == %s", destination.c_str()),
                //  FF("\tfinalfile   == %s", finalfile.c_str()));
                return false;
            }
            
            return true;
        }
        
        PList::Dictionary plist_load(std::string const& source) {
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
        
        void plist_impl(PList::Dictionary& dict, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            std::for_each(dict.Begin(), dict.End(),
                      [&](auto const& kv) {
                            stringmap_ptr->set(
                                kv.first,
                                static_cast<PList::String*>(kv.second)->GetValue());
            });
        }
        
    } /// namespace detail
    
} /// namespace store