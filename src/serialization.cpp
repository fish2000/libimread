/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <sstream>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/pystring.hh>
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
    
    #pragma mark -
    #pragma mark serialization helper implementations
    
    namespace detail {
        
        #pragma mark -
        #pragma mark INI-file serialization helpers
        
        static const std::string secname("yo-dogg");
        
        std::string ini_dumps(store::stringmapper::stringmap_t const& cache) {
            inicpp::config inimap;
            inimap.add_section(secname);
            for (auto const& item : cache) {
                inimap.add_option<inicpp::string_ini_t>(secname, item.first,
                                                                 item.second);
            }
            std::ostringstream stringerator;
            stringerator << inimap;
            return stringerator.str();
        }
        
        void ini_impl(std::string const& inistr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            inicpp::config inimap = inicpp::parser::load(inistr);
            for (auto const& option : inimap[secname]) {
                stringmap_ptr->set(option.get_name(),
                                   option.get<inicpp::string_ini_t>());
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
        
        std::string plist_dumps(store::stringmapper::stringmap_t const& cache) {
            PList::Dictionary dict;
            for (auto const& item : cache) {
                dict.Set(item.first, PList::String(item.second));
            }
            return dict.ToXml();
        }
        
        void plist_impl(std::string const& xmlstr, store::stringmapper* stringmap_ptr) {
            if (stringmap_ptr == nullptr) { return; }           /// `stringmap_ptr` must be a valid pointer
            PList::Dictionary dict = PList::Dictionary::FromXml(xmlstr);
            std::for_each(dict.Begin(), dict.End(),
                      [&](auto const& item) {
                            stringmap_ptr->set(item.first,
                   static_cast<PList::String*>(item.second)->GetValue());
            });
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
        
        store::stringmapper::formatter for_path(std::string const& pth) {
            using filesystem::path;
            std::string ext = pystring::lower(path::extension(pth));
            if (ext == "json") {
                return stringmapper::formatter::json;
            } else if (ext == "plist") {
                return stringmapper::formatter::plist;
            } else if (ext == "pkl" || ext == "pickle") {
                return stringmapper::formatter::pickle;
            } else if (ext == "ini") {
                return stringmapper::formatter::ini;
            } else if (ext == "yml" || ext == "yaml") {
                return stringmapper::formatter::yaml;
            }
            return stringmapper::default_format; /// JSON
        }
        
        std::string string_load(std::string const& source) {
            using filesystem::path;
            
            if (!path::exists(source)) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): nonexistant source file",
                 FF("\tsource == %s", source.c_str()));
            }
            if (!path::is_file_or_link(source)) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): non-file-or-link source file",
                 FF("\tsource == %s", source.c_str()));
            }
            if (!path::is_readable(source)) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): unreadable source file",
                 FF("\tsource == %s", source.c_str()));
            }
            
            std::fstream stream;
            stream.open(source, std::ios::in);
            if (!stream.is_open()) {
                imread_raise(CannotReadError,
                    "store::detail::string_load(source): couldn't open a stream to read source file",
                 FF("\tsource == %s", source.c_str()));
            }
            
            /// Adapted from https://stackoverflow.com/a/3203502/298171
            std::string out(std::istreambuf_iterator<char>(stream), {});
            stream.close();
            
            /// return by value
            return out;
        }
        
        bool string_dump(std::string const& content, std::string const& dest, bool overwrite) {
            using filesystem::path;
            using filesystem::NamedTemporaryFile;
            std::string destination(dest);
            std::string ext;
            
            try {
                destination = path::expand_user(dest).make_absolute().str();
            } catch (im::FileSystemError&) {
                return false;
            }
            
            ext = pystring::lower(path::extension(destination));
            if (ext == "") { ext = "tmp"; }
            
            if (path::exists(destination)) {
                if (!overwrite) {
                    return false;
                } else if (path::is_directory(destination)) {
                    return false;
                }
            }
            
            NamedTemporaryFile tf("." + ext);
            tf.open();
            tf.stream << content;
            tf.close();
            
            if (path::exists(destination)) {
                path::remove(destination);
            }
            
            path finalfile = tf.filepath.duplicate(destination);
            if (!finalfile.is_file()) {
                return false;
            }
            
            return true;
        }
        
    } /// namespace detail
    
} /// namespace store