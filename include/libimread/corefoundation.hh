/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREFOUNDATION_HH_
#define LIBIMREAD_COREFOUNDATION_HH_

#include <memory>
#include <cstdlib>
#include <libimread/store.hh>

#import  <CoreFoundation/CoreFoundation.h>

#define CF_IDX(x)       static_cast<CFIndex>(x)

CF_EXPORT
__attribute__((cf_returns_retained))
CFStringRef CFStringCreateWithSTLString(CFAllocatorRef alloc,
                                        char const* cStr,
                                        CFStringEncoding encoding = kCFStringEncodingUTF8);

CF_EXPORT
__attribute__((cf_returns_retained))
CFStringRef CFStringCreateWithSTLString(CFAllocatorRef alloc,
                                        std::string const& stlStr,
                                        CFStringEncoding encoding = kCFStringEncodingUTF8);

CF_EXPORT
std::string CFStringGetSTLString(CFStringRef theString,
                                 CFStringEncoding encoding = kCFStringEncodingUTF8);

namespace im {
    
    namespace detail {
        
        using pixbuf_t = std::unique_ptr<uint32_t[],
                         std::default_delete<uint32_t[]>>;
        
        template <typename CoreFoundationType>
        struct cfreleaser {
            constexpr cfreleaser() noexcept = default;
            template <typename U> cfreleaser(cfreleaser<U> const&) noexcept {};
            void operator()(CoreFoundationType __attribute__((cf_consumed)) cfp) {
                if (cfp) { CFRelease(cfp); cfp = nullptr; }
            }
        };
        
        template <typename CoreFoundationType>
        using cfp_t = std::unique_ptr<
                      typename std::decay_t<
                               std::remove_pointer_t<CoreFoundationType>>,
                                          cfreleaser<CoreFoundationType>>;
        
        template <typename S> inline
        static void initialize_stringmap(S&& stringmap) {
            static_assert(store::is_stringmapper_v<S>,
                         "im::detail::initialize_stringmap(…) operand must derive from store::stringmapper");
            std::forward<S>(stringmap).set("CFBundleDevelopmentRegion",     "en");
            std::forward<S>(stringmap).set("CFBundleExecutable",            "$(EXECUTABLE_NAME)");
            std::forward<S>(stringmap).set("CFBundleIdentifier",            "$(PRODUCT_BUNDLE_IDENTIFIER)");
            std::forward<S>(stringmap).set("CFBundleInfoDictionaryVersion", "6.0");
            std::forward<S>(stringmap).set("CFBundleName",                  "$(PRODUCT_NAME)");
            std::forward<S>(stringmap).set("CFBundlePackageType",           "APPL");
            std::forward<S>(stringmap).set("CFBundleShortVersionString",    "1.0");
            std::forward<S>(stringmap).set("CFBundleVersion",               "1");
            std::forward<S>(stringmap).set("LSMinimumSystemVersion",        "$(MACOSX_DEPLOYMENT_TARGET)");
            std::forward<S>(stringmap).set("LSRequiresIPhoneOS",            "true");
            std::forward<S>(stringmap).set("NSHumanReadableCopyright",      "Copyright © 2017 Alexander Böhn. All rights reserved.");
            std::forward<S>(stringmap).set("NSMainStoryboardFile",          "Main");
            std::forward<S>(stringmap).set("NSPrincipalClass",              "NSApplication");
            std::forward<S>(stringmap).set("UILaunchStoryboardName",        "LaunchScreen");
            std::forward<S>(stringmap).set("UIMainStoryboardFile",          "Main");
        }
        
        template <typename SMMergeType = store::stringmap> inline
        store::stringmap bundlemap(SMMergeType&& mergeMap = SMMergeType{}) {
            static_assert(store::is_stringmapper_v<SMMergeType>,
                         "im::detail::bundlemap(…) operand must derive from store::stringmapper");
            store::stringmap dict;
            initialize_stringmap(dict);
            dict.update(mergeMap);
            return dict;
        }
    }
    
}

namespace store {
    
    using cfmdict_ptr = im::detail::cfp_t<CFMutableDictionaryRef>;
    
    class cfdict : public store::stringmapper {
        
        public:
            DECLARE_STRINGMAPPER_TEMPLATES(cfdict);
        
        public:
            virtual bool can_store() const noexcept override;
        
        public:
            cfdict(void);
            cfdict(cfdict const&);
            cfdict(cfdict&&) noexcept;
            explicit cfdict(CFDictionaryRef);
            virtual ~cfdict();
        
        protected:
            bool has(std::string const&) const;
            bool has(CFStringRef) const;
        
        public:
            std::string& get_force(std::string const&) const;
        
        public:
            /// implementation of the store::stringmapper API,
            /// in terms of the CoreFoundation CFDictionary API
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
        public:
            __attribute__((cf_returns_retained)) CFDictionaryRef dictionary() const;
            __attribute__((cf_returns_retained)) operator CFDictionaryRef() const;
            __attribute__((cf_returns_retained)) CFMutableDictionaryRef mutabledictionary() const;
            __attribute__((cf_returns_retained)) operator CFMutableDictionaryRef() const;
        
        protected:
            mutable cfmdict_ptr instance{ nullptr };
    };
    
}

#endif /// LIBIMREAD_COREFOUNDATION_HH_