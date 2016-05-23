//
//  AppDelegate.m
//  TestDataViewer
//
//  Created by FI$H 2000 on 2/3/16.
//  Copyright © 2016 Objects In Space And Time. All rights reserved.
//

#import "AppDelegate.h"
#import "MasterViewController.h"
#import <libimread/ext/categories/NSURL+IM.hh>

#include <algorithm>
#include <vector>
#include <regex>
#include <libimread/ext/filesystem/path.h>

using filesystem::path;

@interface  AppDelegate ()
@property (nonatomic, strong) IBOutlet MasterViewController* masterViewController;
@property (weak) IBOutlet NSWindow* window;
@end

static char const* kBasedir = "/Users/fish/Dropbox/libimread/tests/";
#define RE_FLAGS std::regex::extended | std::regex::icase

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)aNotification {
    // Insert code here to initialize your application
    /// 1) create the MasterViewController
    self.masterViewController = [[MasterViewController alloc] initWithNibName:@"MasterViewController"
                                                                       bundle:nil];
    /// 2) Add the view controller to the windows' content view
    [self.window.contentView addSubview:self.masterViewController.view];
    self.masterViewController.view.frame = ((NSView*)self.window.contentView).bounds;

    /// 3) populate stuff
    path basedir(kBasedir);
    path datadir(basedir.make_absolute() / "data");
    std::regex re("(jpg|jpeg)$", RE_FLAGS);
    const std::vector<path> jpgs = datadir.list(re);
    NSMutableArray* mutableStuff = [[NSMutableArray alloc] initWithCapacity:(NSUInteger)jpgs.size()];

    std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
        path imagepath = basedir/p;
        NSURL* url = [[NSURL alloc] initFileURLWithFilesystemPath:imagepath];
        if ([url isImage]) {
            [mutableStuff addObject:url];
        }
    });

    self.stuff = [[NSArray alloc] initWithArray:mutableStuff];
}

- (void)applicationWillTerminate:(NSNotification*)aNotification {
    // Insert code here to tear down your application
}



@end