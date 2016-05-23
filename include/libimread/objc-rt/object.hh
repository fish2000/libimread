/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_OBJECT_HH
#define LIBIMREAD_OBJC_RT_OBJECT_HH

#include <sstream>
#include <string>
#include "types.hh"
#include "selector.hh"
#include "message-args.hh"
#include "traits.hh"

namespace objc {
    
    /// wrapper around an objective-c instance
    /// ... FEATURING:
    /// + automatic scoped memory management via RAII through MRC messages
    /// ... plus fine control through inlined retain/release/autorelease methods
    /// + access to wrapped object pointer via operator*()
    /// + boolean selector-response test via operator[](T t) e.g.
    ///
    ///     objc::object<NSYoDogg*> yodogg([[NSYoDogg alloc] init]);
    ///     if (yodogg["iHeardYouRespondTo:"]) {
    ///         [*yodogg iHeardYouRespondTo:argsInYourArgs];
    ///     }
    ///
    /// + convenience methods e.g. yodogg.classname(), yodogg.description(), yodogg.lookup() ...
    /// + inline bridging template e.g. void* asVoid = yodogg.bridge<void*>();
    /// + E-Z static methods for looking shit up in the runtime heiarchy
    
    template <typename OCType>
    struct object {
        #if !__has_feature(objc_arc)
            using pointer_t = typename std::decay_t<std::conditional_t<
                                                    std::is_pointer<OCType>::value, OCType,
                                                                 std::add_pointer_t<OCType>>> __unsafe_unretained;
        #else
            using pointer_t = typename std::decay_t<std::conditional_t<
                                                    std::is_pointer<OCType>::value, OCType,
                                                                 std::add_pointer_t<OCType>>>;
        #endif
        using object_t = typename std::remove_pointer_t<pointer_t>;
        
        static_assert(objc::traits::is_object<pointer_t>::value,
                      "objc::object<OCType> requires an Objective-C object type");
        
        pointer_t self;
        
        explicit object(pointer_t ii)
            :self(ii)
            {
                retain();
            }
        
        object(const object& other)
            :self(other.self)
            {
                retain();
            }
        
        object(object&& other) noexcept
            :self(other.self)
            {
                retain();
            }
        
        virtual ~object() { release(); }
        
        object& operator=(pointer_t other) {
            if ([self isEqual:other] == NO) {
                object(other).swap(*this);
            }
            return *this;
        }
        
        object& operator=(const object& other) {
            if ([self isEqual:other.self] == NO) {
                object(other).swap(*this);
            }
            return *this;
        }
        
        object& operator=(object&& other) noexcept {
            if ([self isEqual:other.self] == NO) {
                object(std::move(other)).swap(*this);
            }
            return *this;
        }
        
        operator pointer_t()   const { return self; }
        pointer_t operator*()  const { return self; }
        pointer_t operator->() const { return self; }
        
        bool operator==(const object& other) const {
            return objc::to_bool([self isEqual:other.self]);
        }
        bool operator!=(const object& other) const {
            return !objc::to_bool([self isEqual:other.self]);
        }
        bool operator==(const pointer_t& other) const {
            return objc::to_bool([self isEqual:other]);
        }
        bool operator!=(const pointer_t& other) const {
            return !objc::to_bool([self isEqual:other]);
        }
        
        template <typename T> inline
        T bridge() { return objc::bridge<T>(self); }
        
        inline bool responds_to(types::selector s) const {
            return objc::to_bool([self respondsToSelector:s]);
        }
        
        #if !__has_feature(objc_arc)
            inline void retain() const      { [self retain]; }
            inline void release() const     { [self release]; }
            inline void autorelease() const { [self autorelease]; }
        #else
            inline void retain() const      {}
            inline void release() const     {}
            inline void autorelease() const {}
        #endif
        
        bool operator[](types::selector s) const       { return responds_to(s); }
        bool operator[](const objc::selector& s) const { return responds_to(s.sel); }
        bool operator[](const char* s) const           { return responds_to(::sel_registerName(s)); }
        bool operator[](const std::string& s) const    { return responds_to(::sel_registerName(s.c_str())); }
        bool operator[](NSString* s) const             { return responds_to(::NSSelectorFromString(s)); }
        bool operator[](CFStringRef s) const           { return responds_to(::NSSelectorFromString(
                                                                        objc::bridge<NSString*>(s))); }
        
        std::size_t hash() const {
            return static_cast<std::size_t>([self hash]);
        }
        
        types::cls getclass() const {
            return [self class];
        }
        
        std::string classname() const {
            return ::class_getName([self class]);
        }
        
        std::string description() const {
            return [[[self class] description] UTF8String];
        }
        
        types::cls lookup() const {
            return ::objc_lookUpClass(::object_getClassName(self));
        }
        
        friend std::ostream& operator<<(std::ostream& os, const object& friendly) {
            return os << "<" << friendly.classname()   << "> "
                      << "(" << friendly.description() << ") "
                      << "[" << std::hex << "0x"
                             << friendly.hash()
                             << std::dec << "]";
        }
        
        void swap(object& other) noexcept {
            using std::swap;
            using objc::swap;
            swap(this->self, other.self);
        }
        
        void swap(pointer_t& other) noexcept {
            using std::swap;
            using objc::swap;
            swap(this->self, other);
        }
        
        private:
            object(void);
    };
    
    using id = objc::object<types::ID>;
    
} /* namespace objc */


#endif /// LIBIMREAD_OBJC_RT_OBJECT_HH