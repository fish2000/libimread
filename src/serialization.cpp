/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <sstream>
#include <algorithm>
#include <numeric>
#include <memory>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/pystring.hh>
#include <libimread/ext/uri.hh>
#include <libimread/serialization.hh>

/// JSON (built-in)
#include <libimread/ext/JSON/json11.h>

/// libplist
#include <plist/plist++.h>

/// yaml-cpp
#include "yaml-cpp/yaml.h"

/// inicpp (bundled in deps/)
#include "inicpp/inicpp.h"

namespace store {
    
    using namespace im;
    
    #pragma mark -
    #pragma mark serialization helper implementations
    
    namespace detail {
        
        #pragma mark -
        #pragma mark INI-file serialization helpers
        
        static const std::string section_name("stringmapper");
        
        std::string ini_dumps(store::stringmapper::stringmap_t const& cache) {
            inicpp::config inimap;
            inimap.add_section(section_name);
            for (auto const& item : cache) {
                inimap.add_option<inicpp::string_ini_t>(section_name, item.first,
                                                                      item.second);
            }
            std::ostringstream stringerator;
            stringerator << inimap;
            return stringerator.str();
        }
        
        void ini_impl(std::string const& inistr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            inicpp::config inimap = inicpp::parser::load(inistr);
            for (auto const& option : inimap[section_name]) {
                stringmap_ptr->set(option.get_name(),
                              join(option.get_list<inicpp::string_ini_t>(), ", "));
            }
        }
        
        #pragma mark -
        #pragma mark JSON serialization helpers
        
        std::string json_dumps(store::stringmapper::stringmap_t const& cache) {
            return Json(cache).format();
        }
        
        static void json_map_impl(Json const& jsonmap, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr)       { return; }     /// `stringmap_ptr` must be a valid pointer
            if (jsonmap.type() != Type::OBJECT) { return; }     /// `jsonmap` must be a JSON map (née “Object”)
            for (std::string const& key : jsonmap.keys()) {
                stringmap_ptr->set(key, static_cast<std::string>(jsonmap.get(key)));
            }
        }
        
        void json_impl(std::string const& jsonstr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            Json json = Json::parse(jsonstr);
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
        
        using plist_dictptr_t = std::unique_ptr<PList::Dictionary>;
        
        std::string plist_dumps(store::stringmapper::stringmap_t const& cache) {
            PList::Dictionary dict;
            for (auto const& item : cache) {
                dict.Set(item.first, PList::String(item.second));
            }
            return dict.ToXml();
        }
        
        void plist_impl(std::string const& xmlstr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            plist_dictptr_t dict(static_cast<PList::Dictionary*>(
                                             PList::Structure::FromXml(xmlstr)));
            if (!dict.get()) { return; }                        /// `xmlstr` must be a valid XML plist
            std::for_each(dict->Begin(),
                          dict->End(),
                      [&](auto const& item) {
                            stringmap_ptr->set(item.first,
                   static_cast<PList::String*>(item.second)->GetValue()); });
        }
        
        #pragma mark -
        #pragma mark URL parameter serialization helpers
        
        std::string urlparam_dumps(store::stringmapper::stringmap_t const& cache, bool questionmark) {
            store::stringmapper::stringvec_t pairs;
            std::transform(cache.begin(),
                           cache.end(),
                           std::back_inserter(pairs),
                        [](auto const& item) { return uri::encode(item.first)
                                                    + "="
                                                    + uri::encode(item.second); });
            return questionmark ? "?" + join(pairs, "&")
                                      : join(pairs, "&");
        }
        
        void urlparam_impl(std::string const& urlstr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            if (urlstr.size() == 0)       { return; }           /// `urlstr` cannot be zero-length
            std::string url = urlstr[0] == '?' ? std::string(urlstr.begin() + 1,
                                                             urlstr.end())
                                               : std::string(urlstr);
            if (url.size() == 0)          { return; }           /// `url` cannot be zero-length
            store::stringmapper::stringvec_t pairs, pairvec;
            pystring::split(url, pairs, "&");
            for (std::string const& pair : pairs) {
                pystring::split(pair, pairvec, "=");
                switch (pairvec.size()) {
                    case 0: continue;
                    case 1: {
                        stringmap_ptr->set(uri::decode(pairvec[0]),
                                           "true");
                        continue;
                    }
                    case 2:
                    default: {
                        stringmap_ptr->set(uri::decode(pairvec[0]),
                                           uri::decode(pairvec[1]));
                        continue;
                    }
                }
            }
        }
        
        #pragma mark -
        #pragma mark YAML serialization helpers
        
        std::string yaml_dumps(store::stringmapper::stringmap_t const& cache) {
            YAML::Emitter yamitter;
            yamitter.SetIndent(4);
            yamitter << YAML::BeginMap;
            for (auto const& item : cache) {
                yamitter << YAML::Key << item.first;
                yamitter << YAML::Value << item.second;
            }
            yamitter << YAML::EndMap;
            return yamitter.c_str();
        }
        
        void yaml_impl(std::string const& yamlstr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            YAML::Node yaml = YAML::Load(yamlstr);
            for (auto const& item : yaml) {
                stringmap_ptr->set(item.first.as<std::string>(),
                                   item.second.as<std::string>());
            }
        }
        
        #pragma mark -
        #pragma mark Miscellaneous serialization helper functions
        
        std::string join(store::stringmapper::stringvec_t const& strings, std::string const& with) {
            /// Join a vector of strings using reduce (left-fold) as per std::accumulate(…):
            return std::accumulate(strings.begin(),
                                   strings.end(),
                                   std::string{},
                               [&](std::string const& lhs,
                                   std::string const& rhs) {
                return lhs + rhs + (rhs.c_str() == strings.back().c_str() ? "" : with);
            });
        }
        
        store::stringmapper::formatter for_path(std::string const& pth) {
            /// Return a formatter type (q.v the store::stringmapper:formatter enum)
            /// appropriate for the file extension of a given filepath string:
            using filesystem::path;
            std::string ext = pystring::lower(path::extension(pth));
            if (ext == "json") {
                return stringmapper::formatter::json;
            } else if (ext == "plist") {
                return stringmapper::formatter::plist;
            } else if (ext == "pkl" || ext == "pickle") {
                return stringmapper::formatter::pickle;
            } else if (ext == "ini" || ext == "cfg") {
                return stringmapper::formatter::ini;
            } else if (ext == "url" || ext == "uri" ||
                       ext == "urlparam"            ||
                       ext == "uriparam") {
                return stringmapper::formatter::urlparam;
            } else if (ext == "yml" || ext == "yaml") {
                return stringmapper::formatter::yaml;
            }
            return stringmapper::default_format; /// JSON
        }
        
        std::string load(std::string const& source) {
            using filesystem::path;
            path sourcepath(source);
            
            /// Examine the source filepath, and ensure
            /// that it is properly dealt with, depending
            /// on whether it already exists, and how this
            /// function was called:
            if (!sourcepath.exists()) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): nonexistant source file",
                 FF("\tsource == %s", sourcepath.c_str()));
            }
            if (!sourcepath.is_file_or_link()) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): non-file-or-link source file",
                 FF("\tsource == %s", sourcepath.c_str()));
            }
            if (!sourcepath.is_readable()) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): unreadable source file",
                 FF("\tsource == %s", sourcepath.c_str()));
            }
            
            /// Open a file stream for reading,
            /// given the source filepath, and raising
            /// an error if the open attempt fails:
            std::fstream stream;
            stream.open(sourcepath.str(), std::ios::in);
            if (!stream.is_open()) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): couldn't open a stream to read source file",
                 FF("\tsource == %s", sourcepath.c_str()));
            }
            
            /// Read in the contents of the file, to a string, via the stream --
            /// This was adapted from https://stackoverflow.com/a/3203502/298171
            std::string out(std::istreambuf_iterator<char>(stream), {});
            stream.close();
            
            /// Return the populated string by value:
            return out;
        }
        
        bool dump(std::string const& content, std::string const& dest, bool overwrite) {
            using filesystem::path;
            using filesystem::NamedTemporaryFile;
            std::string destination(dest);
            std::string ext;
            path destpath;
            
            /// Properly “normalize” the destination filepath:
            try {
                destpath = path::expand_user(dest).make_absolute();
            } catch (im::FileSystemError&) {
                return false;
            }
            
            /// Divine an appropriate extension for the file,
            /// based on the putative destination filepath:
            ext = pystring::lower(destpath.extension());
            if (ext == "") { ext = "tmp"; }
            
            /// Ensure that the destination filepath is
            /// properly delt with, depending on whether
            /// it exists and how this function was called:
            if (destpath.exists()) {
                if (!overwrite) {
                    return false;
                } else if (destpath.is_directory()) {
                    return false;
                }
            }
            
            /// Create a temporary file,
            /// open its stream interface,
            /// and add in the string the content:
            NamedTemporaryFile tf("." + ext);
            tf.open();
            tf.stream << content;
            tf.close();
            
            /// At this point, we are clear to remove
            /// any existant files at the destination filepath:
            if (destpath.exists()) {
                destpath.remove();
            }
            
            /// Duplicate the temporary file that holds our content
            /// to the (now all-clear) destination filepath:
            path finalfile = tf.filepath.duplicate(destpath);
            if (!finalfile.is_file()) {
                return false;
            }
            
            return true;
        }
        
    } /// namespace detail
    
} /// namespace store