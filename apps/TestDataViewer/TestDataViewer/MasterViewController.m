//
//  MasterViewController.m
//  TestDataViewer
//
//  Created by FI$H 2000 on 2/3/16.
//  Copyright © 2016 Objects In Space And Time. All rights reserved.
//

#import "MasterViewController.h"

@interface MasterViewController ()

@end

@implementation MasterViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do view setup here.
    NSLog(@"%@ (%li) - %@ %@ %@",
          self.view.identifier,
          (long)self.view.tag,
          self.view.description,
          self.view.className,
          self.view.classDescription);

    
}

@end
