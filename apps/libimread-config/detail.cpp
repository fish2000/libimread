/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/pystring.hh>

#include "detail.hh"

namespace im {
    
    namespace config {
        
        using stringvec_t = std::vector<std::string>;
        using filesystem::path;
        
        namespace detail {
            
            /*
                std::for_each(incvec.begin(), incvec.end(),
                    [&outvec](std::string const& pp) {
                    // outvec.emplace_back(path::absolute(p).c_str());
                    // if (path::exists(p)) {
                    //     outvec.emplace_back(path::real(p).str());
                    // }
                        // path intermediate = path("/") / p.substr(1);
                        
                        // path intermediate(true);
                        // intermediate.set(p);
                        path intermediate = pp;
                        
                        stringvec_t parts = intermediate.components();
                        
                        std::string joined = std::accumulate(parts.begin(),
                                                             parts.end(),
                                                             std::string{},
                                                    [&parts](std::string const& lhs,
                                                             std::string const& rhs) {
                            return lhs + rhs + (rhs.c_str() == parts.back().c_str() ? "" : ", ");
                        });
                        
                        WTF("PATH:",
                            FF("string: %s", pp.c_str()),
                            FF("inpath: %s", intermediate.str().c_str()),
                            FF("cparts: %u", parts.size()),
                            FF("joined: %s", joined.c_str()));
                        
                        // outvec.emplace_back(intermediate.str());
                        outvec.push_back(pp);
                });
            */
            
            std::string get_includes(std::string const& inclist) {
                stringvec_t incvec, outvec;
                pystring::split(
                    pystring::replace(inclist,
                                      path(IM_CMAKE_BINARY_DIR).parent().str(),
                                      IM_INSTALL_PREFIX),
                                      incvec, ";");
                std::sort(incvec.begin(), incvec.end());
                auto last = std::unique(incvec.begin(), incvec.end());
                incvec.erase(last, incvec.end());
                std::reverse(incvec.begin(), incvec.end());
                outvec.reserve(incvec.size());
                std::for_each(incvec.begin(), incvec.end(),
                    [&outvec](std::string const& p) {
                    if (pystring::endswith(p, "include") && path::is_directory(p)) {
                        outvec.emplace_back(path::real(p).str());
                    } else {
                        outvec.emplace_back(p);
                    }
                });
                
                std::string yodogg = IM_CMAKE_BINARY_DIR;
                bool does_it_end = pystring::endswith(yodogg, "build");
                bool well_is_it = path::is_directory(yodogg);
                std::string yodad = path::parent(yodogg).str();
                
                // WTF("PARENT PATH:", yodad, yodogg,
                //     does_it_end ? "ends with build" : "does not end with build",
                //     well_is_it  ? "directory" : "not a directory");
                
                return "-I" + pystring::join(" -I", outvec);
            }
            
            std::string get_libs(std::string const& liblist) {
                stringvec_t libvec, outvec;
                pystring::split(
                    pystring::replace(liblist,
                                      path(IM_CMAKE_BINARY_DIR).parent().str(),
                                      IM_INSTALL_PREFIX),
                                      libvec, ";");
                std::sort(libvec.begin(), libvec.end());
                auto last = std::unique(libvec.begin(), libvec.end());
                libvec.erase(last, libvec.end());
                std::reverse(libvec.begin(), libvec.end());
                outvec.reserve(libvec.size());
                std::for_each(libvec.begin(), libvec.end(),
                    [&outvec](std::string const& p) {
                    /// N.B. this resolves library-versioning symlinks,
                    /// which may be undesirable:
                    if (pystring::endswith(p, IM_DYLIB_SUFFIX) && path::is_file(p)) {
                        outvec.emplace_back(path::real(p).str());
                    } else {
                        outvec.emplace_back(p);
                    }
                });
                return "-l" + pystring::join(" -l", outvec);
            }
            
        }
        
    } /* namespace config */
    
} /* namespace im */
