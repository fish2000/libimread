
#include <vector>
#include <memory>
#include <unordered_map>
#import  <Foundation/Foundation.h>
#import  <AppKit/AppKit.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/objc-rt/objc-rt.hh>
#include <libimread/objc-rt/appkit.hh>
#include <libimread/ext/filesystem/path.h>
#import  <libimread/ext/categories/NSBitmapImageRep+IM.hh>
#import  <libimread/ext/categories/NSURL+IM.hh>
#include <libimread/image.hh>
// #include <libimread/interleaved.hh>
#include <libimread/halide.hh>
#include <libimread/hashing.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using im::byte;
    using im::Image;
    // using im::InterleavedFactory;
    using im::HalideFactory;
    
    TEST_CASE("[appkit-copy-paste] Copy and paste PNG image data",
              "[appkit-copy-paste-png-image-data]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        
        @autoreleasepool {
            
            std::vector<NSPasteboard*> boards;
            std::unordered_map<path, objc::object<NSImage>> images;
            
            std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
                path imagepath = basedir/p;
                NSPasteboard* board = [NSPasteboard pasteboardWithUniqueName];
                boards.push_back(board);
                
                NSURL* url = [[NSURL alloc] initFileURLWithFilesystemPath:imagepath];
                NSImage* image = [[NSImage alloc] initWithContentsOfURL:url];
                images.insert({ imagepath, objc::object<NSImage>(image) });
                
                BOOL copied = objc::appkit::copy_to(board, image, url);
                CHECK(objc::to_bool(copied));
            });
            
            std::for_each(boards.begin(), boards.end(), [&](NSPasteboard* board) {
                BOOL can_paste_url = objc::appkit::can_paste<NSURL>(board);
                BOOL can_paste_image = objc::appkit::can_paste<NSImage>(board);
                CHECK(objc::to_bool(can_paste_url));
                CHECK(objc::to_bool(can_paste_image));
                
                NSURL* url = objc::appkit::paste<NSURL>(board);
                NSImage* boardimage = objc::appkit::paste<NSImage>(board);
                NSImage* mapimage = images.at([url filesystemPath]);
                
                // InterleavedFactory factory;
                HalideFactory<byte> factory;
                std::hash<Image> hasher;
                std::unique_ptr<Image> boardim;
                std::unique_ptr<Image> mapim;
                NSBitmapImageRep* boardrep;
                NSBitmapImageRep* maprep;
                
                for (id rep in boardimage.representations) {
                    if ([rep isKindOfClass:[NSBitmapImageRep class]]) {
                        boardrep = (NSBitmapImageRep*)rep;
                        boardim = [boardrep imageUsingImageFactory:&factory];
                        break;
                    }
                }
                for (id rep in mapimage.representations) {
                    if ([rep isKindOfClass:[NSBitmapImageRep class]]) {
                        maprep = (NSBitmapImageRep*)rep;
                        mapim = [maprep imageUsingImageFactory:&factory];
                        break;
                    }
                }
                
                CHECK(boardim->size() > 0);
                CHECK(mapim->size() > 0);
                CHECK(boardim->size() == mapim->size());
                
                auto boardhash = blockhash::blockhash(*boardim).to_ulong();
                auto maphash = blockhash::blockhash(*mapim).to_ulong();
                
                // CHECK(objc::to_bool([boardimage isEqual:mapimage]));
                // CHECK(hasher(*boardim) == hasher(*mapim));
                CHECK(boardhash == maphash);
                
            });
            
            std::for_each(boards.begin(), boards.end(), [&](NSPasteboard* board) {
                [board releaseGlobally];
            });
            
        }
        
    }
    
    
    TEST_CASE("[appkit-copy-paste] Copy and paste JPEG image data",
              "[appkit-copy-paste-jpeg-image-data]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        @autoreleasepool {
            
            std::vector<NSPasteboard*> boards;
            std::unordered_map<path, objc::object<NSImage>> images;
            
            std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
                path imagepath = basedir/p;
                NSPasteboard* board = [NSPasteboard pasteboardWithUniqueName];
                boards.push_back(board);
                
                NSURL* url = [[NSURL alloc] initFileURLWithFilesystemPath:imagepath];
                NSImage* image = [[NSImage alloc] initWithContentsOfURL:url];
                images.insert({ imagepath, objc::object<NSImage>(image) });
                
                BOOL copied = objc::appkit::copy_to(board, image, url);
                CHECK(objc::to_bool(copied));
            });
            
            std::for_each(boards.begin(), boards.end(), [&](NSPasteboard* board) {
                BOOL can_paste_url = objc::appkit::can_paste<NSURL>(board);
                BOOL can_paste_image = objc::appkit::can_paste<NSImage>(board);
                CHECK(objc::to_bool(can_paste_url));
                CHECK(objc::to_bool(can_paste_image));
                
                NSURL* url = objc::appkit::paste<NSURL>(board);
                NSImage* boardimage = objc::appkit::paste<NSImage>(board);
                NSImage* mapimage = images.at([url filesystemPath]);
                
                // InterleavedFactory factory;
                // HalideFactory<byte> factory;
                // std::hash<Image> hasher;
                // std::unique_ptr<Image> boardim;
                // std::unique_ptr<Image> mapim;
                // NSBitmapImageRep* boardrep;
                // NSBitmapImageRep* maprep;
                NSData* boarddata = [boardimage TIFFRepresentation];
                NSData* mapdata = [mapimage TIFFRepresentation];
                
                CHECK(objc::to_bool([boarddata isEqualToData:mapdata]));
                
                // for (id rep in boardimage.representations) {
                //     if ([rep isKindOfClass:[NSBitmapImageRep class]]) {
                //         boardrep = (NSBitmapImageRep*)rep;
                //         boardim = [boardrep imageUsingImageFactory:&factory];
                //         break;
                //     }
                // }
                // for (id rep in mapimage.representations) {
                //     if ([rep isKindOfClass:[NSBitmapImageRep class]]) {
                //         maprep = (NSBitmapImageRep*)rep;
                //         mapim = [maprep imageUsingImageFactory:&factory];
                //         break;
                //     }
                // }
                
                // CHECK(boardim->size() > 0);
                // CHECK(mapim->size() > 0);
                // CHECK(boardim->size() == mapim->size());
                //
                // auto boardhash = blockhash::blockhash(*boardim).to_ulong();
                // auto maphash = blockhash::blockhash(*mapim).to_ulong();
                
                // CHECK(objc::to_bool([boardimage isEqual:mapimage]));
                // CHECK(hasher(*boardim) == hasher(*mapim));
                // CHECK(boardhash == maphash);
                
            });
            
            std::for_each(boards.begin(), boards.end(), [&](NSPasteboard* board) {
                [board releaseGlobally];
            });
            
        }
        
    }
    
}

