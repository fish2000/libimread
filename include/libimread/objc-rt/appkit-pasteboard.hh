/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_APPKIT_PASTEBOARD_HH
#define LIBIMREAD_OBJC_RT_APPKIT_PASTEBOARD_HH

#include <unordered_map>
#include <unordered_set>

#include <libimread/ext/pystring.hh>
#include <libimread/ext/filesystem/path.h>
#import  <libimread/ext/categories/NSString+STL.hh>
#import  <libimread/ext/categories/NSURL+IM.hh>
#include "objc-rt.hh"
#include "appkit.hh"

namespace objc {
    
    namespace appkit {
        
        struct PasteboardSubBase : public objc::object<NSPasteboard> {
            using pointer_t = objc::nonnull_ptr_t<objc::object<NSPasteboard>::pointer_t>;
            using object_t = objc::object<NSPasteboard>::object_t;
            using typemap_t = std::unordered_map<std::string, objc::nonnull_ptr_t<NSString>>;
            using typepam_t = std::unordered_map<objc::nonnull_ptr_t<NSString>, std::string>;
            using typeset_t = std::unordered_set<std::string>;
            using stringarray_t = objc::nonnull_ptr_t<NSArray<NSString*>>;
            
            static const typemap_t typemap;
            static const typepam_t typepam;
            static typemap_t init_typemap() {
                typemap_t _typemap = {
                    { "string",         NSPasteboardTypeString                  },
                    { "pdf",            NSPasteboardTypePDF                     },
                    { "tiff",           NSPasteboardTypeTIFF                    },
                    { "png",            NSPasteboardTypePNG                     },
                    { "rtf",            NSPasteboardTypeRTF                     },
                    { "rtfd",           NSPasteboardTypeRTFD                    },
                    { "html",           NSPasteboardTypeHTML                    },
                    { "tabulartext",    NSPasteboardTypeTabularText             },
                    { "font",           NSPasteboardTypeFont                    },
                    { "ruler",          NSPasteboardTypeRuler                   },
                    { "color",          NSPasteboardTypeColor                   },
                    { "sound",          NSPasteboardTypeSound                   },
                    { "multitext",      NSPasteboardTypeMultipleTextSelection   },
                    { "findopts",       NSPasteboardTypeFindPanelSearchOptions  }
                };
                return _typemap;
            }
            static typepam_t init_typepam() {
                typepam_t _typepam = {
                    { NSPasteboardTypeString,                   "string",       },
                    { NSPasteboardTypePDF,                      "pdf",          },
                    { NSPasteboardTypeTIFF,                     "tiff",         },
                    { NSPasteboardTypePNG,                      "png",          },
                    { NSPasteboardTypeRTF,                      "rtf",          },
                    { NSPasteboardTypeRTFD,                     "rtfd",         },
                    { NSPasteboardTypeHTML,                     "html",         },
                    { NSPasteboardTypeTabularText,              "tabulartext",  },
                    { NSPasteboardTypeFont,                     "font",         },
                    { NSPasteboardTypeRuler,                    "ruler",        },
                    { NSPasteboardTypeColor,                    "color",        },
                    { NSPasteboardTypeSound,                    "sound",        },
                    { NSPasteboardTypeMultipleTextSelection,    "multitext",    },
                    { NSPasteboardTypeFindPanelSearchOptions,   "findopts",     }
                };
                return _typepam;
            }
            
            PasteboardSubBase(pointer_t pointer)
                :objc::object<NSPasteboard>(pointer)
                {}
            
        };
        
        
        template <typename PasteboardClass>
        struct PasteboardBase : public PasteboardSubBase {
            using PasteboardSubBase::pointer_t;
            using PasteboardSubBase::object_t;
            using objc::object<NSPasteboard>::self;
            
            PasteboardBase(pointer_t pointer)
                :PasteboardSubBase(pointer)
                {}
            
            std::string name() const {
                return [self.name STLString];
            }
            
            stringarray_t types() const {
                return self.types;
            }
            
            typeset_t typeset() const {
                typeset_t out;
                auto seterator = out.begin();
                for (objc::nonnull_ptr_t<NSString> type in self.types) {
                    seterator = out.emplace_hint(seterator,
                                                 typepam.at(type));
                }
                return out;
            }
            
            template <typename OCType>
            BOOL can_paste() const noexcept {
                return objc::appkit::can_paste<OCType>(self);
            }
            
            template <typename OCType>
            __attribute__((ns_returns_retained))
            typename std::add_pointer_t<OCType> _Nullable paste() const noexcept {
                return objc::appkit::paste<OCType>(self);
            }
            
            template <typename ...OCTypes> inline
            BOOL copy(OCTypes _Nonnull... objects) noexcept {
                return objc::appkit::copy_to<OCTypes...>(self, objects...);
            }
            
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wnullability-completeness"
            PasteboardClass expand() {
                /// I found the warnings about this function returning
                /// a potentially null pointer to be really funny
                /// -- but only the first time
                pointer_t newSelf = [object_t pasteboardByFilteringTypesInPasteboard:self];
                if (objc::to_bool([self isEqual:newSelf])) { return *this; }
                return PasteboardClass(newSelf);
            }
            #pragma clang diagnostic pop
        };
        
        class GeneralPasteboard : public PasteboardBase<GeneralPasteboard> {
            
            public:
                
                using Base = PasteboardBase<GeneralPasteboard>;
                using Base::self;
                
                GeneralPasteboard()
                    :Base(
                        [NSPasteboard generalPasteboard])
                    {}
            
        };
        
        class Pasteboard : public PasteboardBase<Pasteboard> {
            
            public:
                
                using Base = PasteboardBase<Pasteboard>;
                using Base::self;
                
                Pasteboard()
                    :Base(
                        [NSPasteboard pasteboardWithUniqueName])
                    {}
                
                Pasteboard(objc::nonnull_ptr_t<NSString> name)
                    :Base(
                        [NSPasteboard pasteboardWithName:name])
                    {}
                Pasteboard(std::string const& name)
                    :Base(
                        [NSPasteboard pasteboardWithName:[NSString stringWithSTLString:name]])
                    {}
                
                ~Pasteboard() {
                    [self releaseGlobally];
                }
            
        };
            
        class FileSource : public PasteboardBase<FileSource> {
            
            public:
                
                using Base = PasteboardBase<FileSource>;
                using Base::self;
                
                FileSource(objc::nonnull_ptr_t<NSURL> url)
                    :Base(
                        [NSPasteboard pasteboardByFilteringFile:url.path])
                    {}
                FileSource(filesystem::path const& pth)
                    :Base(
                        [NSPasteboard pasteboardByFilteringFile:[NSString stringWithSTLString:pth.str()]])
                    {}
                
                ~FileSource() {
                    [self releaseGlobally];
                }
            
        };
        
        class DataSource : public PasteboardBase<DataSource> {
            
            public:
                using Base = PasteboardBase<DataSource>;
                using Base::self;
                using Base::typemap;
                
                DataSource(objc::nonnull_ptr_t<NSData> datum,
                           objc::nonnull_ptr_t<NSString> type)
                    :Base(
                        [NSPasteboard pasteboardByFilteringData:datum
                                                         ofType:type])
                    {}
                DataSource(objc::nonnull_ptr_t<NSData> datum,
                           std::string const& type)
                    :Base(
                        [NSPasteboard pasteboardByFilteringData:datum
                                                         ofType:objc::nonnull_cast(
                                                                typemap.at(pystring::lower(type)))])
                    {}
                
                ~DataSource() {
                    [self releaseGlobally];
                }
            
        };
        
        
    }
    
}

#endif /// LIBIMREAD_OBJC_RT_APPKIT_PASTEBOARD_HH