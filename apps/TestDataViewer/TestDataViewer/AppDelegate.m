//
//  AppDelegate.m
//  TestDataViewer
//
//  Created by FI$H 2000 on 2/3/16.
//  Copyright © 2016 Objects In Space And Time. All rights reserved.
//

#import "AppDelegate.h"
#import "MasterViewController.h"

@interface  AppDelegate ()
@property (nonatomic, strong) IBOutlet MasterViewController *masterViewController;
@end

@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // Insert code here to initialize your application
    /// 1) create the MasterViewController
    self.masterViewController = [[MasterViewController alloc] initWithNibName:@"MasterViewController"
                                                                       bundle:nil];
    /// 2) Add the view controller to the windows' content view
    [self.window.contentView addSubview:self.masterViewController.view];
    self.masterViewController.view.frame = ((NSView*)self.window.contentView).bounds;
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    // Insert code here to tear down your application
}

@end
