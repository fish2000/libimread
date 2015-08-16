/*
 * Copyright (c) 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015
 *   Jonathan Schleifer <js@webkeks.org>
 *
 * All rights reserved.
 *
 * This file is part of ObjFW. It may be distributed under the terms of the
 * Q Public License 1.0, which can be found in the file LICENSE.QPL included in
 * the packaging of this file.
 *
 * Alternatively, it may be distributed under the terms of the GNU General
 * Public License, either version 2 or 3, which can be found in the file
 * LICENSE.GPLv2 or LICENSE.GPLv3 respectively included in the packaging of this
 * file.
 */

#include "config.h"

#import "OFThread.h"
#import "OFString.h"
#import "OFDictionary.h"
#import "OFAutoreleasePool.h"

#import "TestsAppDelegate.h"

static OFString *module = @"OFThread";

@interface TestThread: OFThread
@end

@implementation TestThread
- (id)main
{
	[[OFThread threadDictionary] setObject: @"bar"
					forKey: @"foo"];

	return @"success";
}
@end

@implementation TestsAppDelegate (OFThreadTests)
- (void)threadTests
{
	OFAutoreleasePool *pool = [[OFAutoreleasePool alloc] init];
	TestThread *t;
	OFMutableDictionary *d;

	TEST(@"+[thread]", (t = [TestThread thread]))

	TEST(@"-[start]", R([t start]))

	TEST(@"-[join]", [[t join] isEqual: @"success"])

	TEST(@"-[threadDictionary]", (d = [OFThread threadDictionary]) &&
	    [d objectForKey: @"foo"] == nil)

	[pool drain];
}
@end
