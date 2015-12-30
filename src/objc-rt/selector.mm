/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/objc-rt/selector.hh>

namespace objc {
    
    selector::selector(const std::string& name)
        :sel(::sel_registerName(name.c_str()))
        {}
    selector::selector(const char* name)
        :sel(::sel_registerName(name))
        {}
    selector::selector(NSString* name)
        :sel(::NSSelectorFromString(name))
        {}
    
    selector::selector(types::selector s)
        :sel(s)
        {}
    selector::selector(const objc::selector& other)
        :sel(other.sel)
        {}
    selector::selector(objc::selector&& other) noexcept
        :sel(other.sel)
        {}
    
    objc::selector& selector::operator=(const objc::selector& other) {
        objc::selector(other).swap(*this);
        return *this;
    }
    objc::selector& selector::operator=(types::selector other) {
        objc::selector(other).swap(*this);
        return *this;
    }
    
    bool selector::operator==(const objc::selector& s) const {
        return objc::to_bool(::sel_isEqual(sel, s.sel));
    }
    bool selector::operator!=(const objc::selector& s) const {
        return !objc::to_bool(::sel_isEqual(sel, s.sel));
    }
    bool selector::operator==(const types::selector& s) const {
        return objc::to_bool(::sel_isEqual(sel, s));
    }
    bool selector::operator!=(const types::selector& s) const {
        return !objc::to_bool(::sel_isEqual(sel, s));
    }
    
    const char* selector::c_str() const {
        return ::sel_getName(sel);
    }
    std::string selector::str() const {
        return c_str();
    }
    NSString* selector::ns_str() const {
        return ::NSStringFromSelector(sel);
    }
    CFStringRef selector::cf_str() const {
        return objc::bridge<CFStringRef>(ns_str());
    }
    
    static std::hash<std::string> stringhasher;
    
    std::size_t selector::hash() const {
        return stringhasher(str());
    }
    
    void selector::swap(objc::selector& other) noexcept {
        using std::swap;
        using objc::swap;
        swap(this->sel, other.sel);
    }
    void selector::swap(types::selector& other) noexcept {
        using std::swap;
        using objc::swap;
        swap(this->sel, other);
    }
    
    selector::operator types::selector() const { return sel; }
    selector::operator std::string() const { return str(); }
    selector::operator const char*() const { return c_str(); }
    selector::operator char*() const { return const_cast<char*>(c_str()); }
    selector::operator NSString*() const { return ::NSStringFromSelector(sel); }
    selector::operator CFStringRef() const { return objc::bridge<CFStringRef>(ns_str()); }

}

objc::selector operator"" _SEL(const char* name) {
    return objc::selector(name);
}

