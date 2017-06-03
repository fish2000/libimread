/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_LIBIMREAD_CONFIG_H_
#define APPS_LIBIMREAD_CONFIG_H_

#include <string>
#include <vector>
#include <algorithm>
#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/pystring.hh>

namespace im {
    
    namespace config {
        
        using stringvec_t = std::vector<std::string>;
        using filesystem::path;
        
        namespace {
            
            inline std::string get_includes() {
                stringvec_t incvec, outvec;
                pystring::split(pystring::replace(IM_INCLUDE_DIRECTORIES,
                                                  IM_CMAKE_BINARY_DIR,
                                                  IM_INSTALL_PREFIX), incvec, ";");
                std::sort(incvec.begin(), incvec.end());
                auto last = std::unique(incvec.begin(), incvec.end());
                incvec.erase(last, incvec.end());
                std::reverse(incvec.begin(), incvec.end());
                std::for_each(incvec.begin(), incvec.end(),
                    [&outvec](std::string const& p) { outvec.emplace_back(path::real(p).str()); });
                return "-I" + pystring::join(" -I", outvec);
            }
            
            inline std::string get_libs() {
                stringvec_t libvec, outvec;
                pystring::split(pystring::replace(IM_LINK_LIBRARIES,
                                                  IM_CMAKE_BINARY_DIR,
                                                  IM_INSTALL_PREFIX), libvec, ";");
                std::sort(libvec.begin(), libvec.end());
                auto last = std::unique(libvec.begin(), libvec.end());
                libvec.erase(last, libvec.end());
                std::reverse(libvec.begin(), libvec.end());
                std::for_each(libvec.begin(), libvec.end(),
                    [&outvec](std::string const& p) {
                        /// N.B. this resolves library-versioning symlinks,
                        /// which may be undesirable:
                        outvec.emplace_back(pystring::endswith(p, IM_DYLIB_SUFFIX) ? path::real(p).str() : p);
                });
                return "-l" + pystring::join(" -l", outvec);
            }
            
        }
        
        const std::string version = IM_VERSION;
        
        const std::string prefix = std::string(IM_INSTALL_PREFIX);
        const std::string exec_prefix = std::string(IM_INSTALL_PREFIX);
        const std::string includes = get_includes();
        const std::string libs = get_libs();
        
        const std::string cflags = std::string(IM_COMPILE_OPTIONS) + " " + includes;
        const std::string ldflags = std::string(IM_LINK_FLAGS) + " " + libs;
        
    };
    
};


#endif /// APPS_LIBIMREAD_CONFIG_H_