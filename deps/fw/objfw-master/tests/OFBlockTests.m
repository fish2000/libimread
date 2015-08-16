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

#import "OFBlock.h"
#import "OFString.h"
#import "OFAutoreleasePool.h"

#if defined(OF_OBJFW_RUNTIME)
# include "runtime.h"
#elif defined(OF_APPLE_RUNTIME)
# include <objc/runtime.h>
#endif

#import "TestsAppDelegate.h"

static OFString *module = @"OFBlock";

extern void *_NSConcreteStackBlock;
extern void *_NSConcreteGlobalBlock;
extern void *_NSConcreteMallocBlock;

static void (^g)() = ^ {};

@implementation TestsAppDelegate (OFBlockTests)
- (void)blockTests
{
	OFAutoreleasePool *pool = [[OFAutoreleasePool alloc] init];
	__block int x;
	void (^s)() = ^ { x = 0; };
	void (^m)();

	TEST(@"Class of stack block",
	    (Class)&_NSConcreteStackBlock == objc_getClass("OFStackBlock") &&
	    [s isKindOfClass: [OFBlock class]])

	TEST(@"Class of global block",
	    (Class)&_NSConcreteGlobalBlock == objc_getClass("OFGlobalBlock") &&
	    [g isKindOfClass: [OFBlock class]])

	TEST(@"Class of a malloc block",
	    (Class)&_NSConcreteMallocBlock == objc_getClass("OFMallocBlock"))

	TEST(@"Copying a stack block",
	    (m = [[s copy] autorelease]) &&
	    [m class] == objc_getClass("OFMallocBlock") &&
	    [m isKindOfClass: [OFBlock class]])

	TEST(@"Copying a global block", (id)g == [[g copy] autorelease])

	TEST(@"Copying a malloc block",
	    (id)m == [m copy] && [m retainCount] == 2)

	TEST(@"Autorelease a stack block", R([s autorelease]))

	TEST(@"Autorelease a global block", R([g autorelease]))

	TEST(@"Autorelease a malloc block", R([m autorelease]))

	[pool drain];
}
@end
