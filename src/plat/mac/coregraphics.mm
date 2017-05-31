// Copyright 2017 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <libimread/coregraphics.hh>

#define STRINGNULL() store::stringmapper::base_t::null_value()

CFStringRef CFStringCreateWithSTLString(CFAllocatorRef alloc,
                                        char const* cStr,
                                        CFStringEncoding encoding) {
    return CFStringCreateWithCString(alloc, cStr, encoding);
}

CFStringRef CFStringCreateWithSTLString(CFAllocatorRef alloc,
                                        std::string const& stlStr,
                                        CFStringEncoding encoding) {
    return CFStringCreateWithCString(alloc, stlStr.c_str(), encoding);
}

std::string CFStringGetSTLString(CFStringRef theString,
                                 CFStringEncoding encoding) {
    std::string out(CFStringGetCStringPtr(theString, encoding));
    return out;
}

namespace store {
    
    bool cfdict::can_store() const noexcept { return true; }
    
    /// default constructor
    cfdict::cfdict(void)
        :instance{ const_cast<__CFDictionary *>(
                      CFDictionaryCreateMutable(kCFAllocatorDefault, CF_IDX(0),
                                               &kCFCopyStringDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks)) }
        {}
    
    /// copy constructor
    cfdict::cfdict(cfdict const& other)
        :instance{ const_cast<__CFDictionary *>(
                  CFDictionaryCreateMutableCopy(kCFAllocatorDefault,
                                                other.count(),
                                                other.instance.get())) }
        {}
    
    /// move constructor
    cfdict::cfdict(cfdict&& other) noexcept
        :instance(std::move(other.instance))
        {}
    
    cfdict::cfdict(CFDictionaryRef raw)
        :instance{ const_cast<__CFDictionary *>(
                  CFDictionaryCreateMutableCopy(kCFAllocatorDefault, CF_IDX(0), raw)) }
        {}
    
    cfdict::cfdict(CFMutableDictionaryRef raw)
        :instance{ const_cast<__CFDictionary *>(
                  CFDictionaryCreateMutableCopy(kCFAllocatorDefault, CF_IDX(0), raw)) }
        {}
    
    cfdict::~cfdict() {}
    
    bool cfdict::has(std::string const& key) const {
        im::detail::cfp_t<CFStringRef> cfkey(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, key)));
        return CFDictionaryContainsKey(instance.get(), cfkey.get());
    }
    
    bool cfdict::has(CFStringRef cfkey) const {
        return CFDictionaryContainsKey(instance.get(), cfkey);
    }
    
    std::string const& cfdict::get_force(std::string const& key) const {
        im::detail::cfp_t<CFStringRef> cfkey(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, key)));
        if (has(cfkey.get())) {
            im::detail::cfp_t<CFStringRef> cfval(
                       static_cast<__CFString *>(
                              const_cast<void *>(
                            CFDictionaryGetValue(instance.get(),
                                                    cfkey.get()))));
            cache[key] = CFStringGetSTLString(cfval.get());
            return cache.at(key);
        }
        return STRINGNULL();
    }
    
    std::string& cfdict::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        }
        im::detail::cfp_t<CFStringRef> cfkey(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, key)));
        if (has(cfkey.get())) {
            im::detail::cfp_t<CFStringRef> cfval(
                       static_cast<__CFString *>(
                              const_cast<void *>(
                            CFDictionaryGetValue(instance.get(),
                                                    cfkey.get()))));
            cache[key] = CFStringGetSTLString(cfval.get());
            return cache.at(key);
        }
        return STRINGNULL();
    }
    
    std::string const& cfdict::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        }
        im::detail::cfp_t<CFStringRef> cfkey(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, key)));
        if (has(cfkey.get())) {
            im::detail::cfp_t<CFStringRef> cfval(
                       static_cast<__CFString *>(
                              const_cast<void *>(
                            CFDictionaryGetValue(instance.get(),
                                                    cfkey.get()))));
            cache[key] = CFStringGetSTLString(cfval.get());
            return cache.at(key);
        }
        return STRINGNULL();
    }
    
    bool cfdict::set(std::string const& key, std::string const& value) {
        if (value == STRINGNULL()) { return del(key); }
        im::detail::cfp_t<CFStringRef> cfkey(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, key)));
        im::detail::cfp_t<CFStringRef> cfval(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, value)));
        CFDictionarySetValue(instance.get(),
                                cfkey.get(),
                                cfval.get());
        cache[key] = value;
        return true;
    }
    
    bool cfdict::del(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            cache.erase(key);
        }
        im::detail::cfp_t<CFStringRef> cfkey(
                    const_cast<__CFString *>(
                 CFStringCreateWithSTLString(kCFAllocatorDefault, key)));
        if (has(cfkey.get())) {
            CFDictionaryRemoveValue(instance.get(),
                                       cfkey.get());
            return true;
        }
        return false;
    }
    
    std::size_t cfdict::count() const {
        return static_cast<std::size_t>(
                   CFDictionaryGetCount(instance.get()));
    }
    
    cfdict::stringvec_t cfdict::list() const {
        stringvec_t outvec{};
        
        /// Directly calling CFDictionaryGetCount --
        /// rather than cfdict::count() -- saves us
        /// the minute (yet arguably circuitous) cost
        /// of dynamic method-dispatch via vtable...
        /// not to mention whatever register-abuse the
        /// gratuitous return-type casting may incur, dogg.
        outvec.reserve(CFDictionaryGetCount(
                               instance.get()));
        
        /// Remember, a non-capturing lambda can
        /// implicitly convert to a function pointer!
        CFDictionaryApplyFunction(instance.get(),
                               [](const void* key,
                                  const void* value,
                                        void* context) {
             CFStringRef cfkey = static_cast<CFStringRef>(key);
             stringvec_t* vecp = static_cast<stringvec_t*>(context);
             vecp->emplace_back(CFStringGetSTLString(cfkey));
        }, &outvec);
        
        return outvec;
    }
    
    CFDictionaryRef cfdict::cfdictionary() const {
        return CFDictionaryCreateCopy(kCFAllocatorDefault,
                                      instance.get());
    }
    
    cfdict::operator CFDictionaryRef() const {
        return CFDictionaryCreateCopy(kCFAllocatorDefault,
                                      instance.get());
    }
    
    CFMutableDictionaryRef cfdict::cfmutabledictionary() const {
        return CFDictionaryCreateMutableCopy(kCFAllocatorDefault,
                        CFDictionaryGetCount(instance.get()),
                                             instance.get());
    }
    
    cfdict::operator CFMutableDictionaryRef() const {
        return CFDictionaryCreateMutableCopy(kCFAllocatorDefault,
                        CFDictionaryGetCount(instance.get()),
                                             instance.get());
    }
    
}