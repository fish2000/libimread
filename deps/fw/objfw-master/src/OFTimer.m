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

#include <stdlib.h>

#include <assert.h>

#import "OFTimer.h"
#import "OFTimer+Private.h"
#import "OFDate.h"
#import "OFRunLoop.h"
#import "OFRunLoop+Private.h"
#ifdef OF_HAVE_THREADS
# import "OFCondition.h"
#endif

#import "OFInvalidArgumentException.h"

@implementation OFTimer
+ (instancetype)scheduledTimerWithTimeInterval: (of_time_interval_t)timeInterval
					target: (id)target
				      selector: (SEL)selector
				       repeats: (bool)repeats
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					    target: target
					  selector: selector
					   repeats: repeats] autorelease];

	[[OFRunLoop currentRunLoop] addTimer: timer];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}

+ (instancetype)scheduledTimerWithTimeInterval: (of_time_interval_t)timeInterval
					target: (id)target
				      selector: (SEL)selector
					object: (id)object
				       repeats: (bool)repeats
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					    target: target
					  selector: selector
					    object: object
					   repeats: repeats] autorelease];

	[[OFRunLoop currentRunLoop] addTimer: timer];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}

+ (instancetype)scheduledTimerWithTimeInterval: (of_time_interval_t)timeInterval
					target: (id)target
				      selector: (SEL)selector
					object: (id)object1
					object: (id)object2
				       repeats: (bool)repeats
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					    target: target
					  selector: selector
					    object: object1
					    object: object2
					   repeats: repeats] autorelease];

	[[OFRunLoop currentRunLoop] addTimer: timer];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}

#ifdef OF_HAVE_BLOCKS
+ (instancetype)scheduledTimerWithTimeInterval: (of_time_interval_t)timeInterval
				       repeats: (bool)repeats
					 block: (of_timer_block_t)block
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					   repeats: repeats
					     block: block] autorelease];

	[[OFRunLoop currentRunLoop] addTimer: timer];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}
#endif

+ (instancetype)timerWithTimeInterval: (of_time_interval_t)timeInterval
			       target: (id)target
			     selector: (SEL)selector
			      repeats: (bool)repeats
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					    target: target
					  selector: selector
					   repeats: repeats] autorelease];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}

+ (instancetype)timerWithTimeInterval: (of_time_interval_t)timeInterval
			       target: (id)target
			     selector: (SEL)selector
			       object: (id)object
			      repeats: (bool)repeats
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					    target: target
					  selector: selector
					    object: object
					   repeats: repeats] autorelease];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}

+ (instancetype)timerWithTimeInterval: (of_time_interval_t)timeInterval
			       target: (id)target
			     selector: (SEL)selector
			       object: (id)object1
			       object: (id)object2
			      repeats: (bool)repeats
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					    target: target
					  selector: selector
					    object: object1
					    object: object2
					   repeats: repeats] autorelease];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}

#ifdef OF_HAVE_BLOCKS
+ (instancetype)timerWithTimeInterval: (of_time_interval_t)timeInterval
			      repeats: (bool)repeats
				block: (of_timer_block_t)block
{
	void *pool = objc_autoreleasePoolPush();
	OFDate *fireDate = [OFDate dateWithTimeIntervalSinceNow: timeInterval];
	id timer = [[[self alloc] initWithFireDate: fireDate
					  interval: timeInterval
					   repeats: repeats
					     block: block] autorelease];

	[timer retain];
	objc_autoreleasePoolPop(pool);

	return [timer autorelease];
}
#endif

- init
{
	OF_INVALID_INIT_METHOD
}

- (instancetype)OF_initWithFireDate: (OFDate*)fireDate
			   interval: (of_time_interval_t)interval
			     target: (id)target
			   selector: (SEL)selector
			     object: (id)object1
			     object: (id)object2
			  arguments: (uint8_t)arguments
			    repeats: (bool)repeats
{
	self = [super init];

	@try {
		_fireDate = [fireDate retain];
		_interval = interval;
		_target = [target retain];
		_selector = selector;
		_object1 = [object1 retain];
		_object2 = [object2 retain];
		_arguments = arguments;
		_repeats = repeats;
		_valid = true;
#ifdef OF_HAVE_THREADS
		_condition = [[OFCondition alloc] init];
#endif
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithFireDate: (OFDate*)fireDate
	  interval: (of_time_interval_t)interval
	    target: (id)target
	  selector: (SEL)selector
	   repeats: (bool)repeats
{
	return [self OF_initWithFireDate: fireDate
				interval: interval
				  target: target
				selector: selector
				  object: nil
				  object: nil
			       arguments: 0
				 repeats: repeats];
}

- initWithFireDate: (OFDate*)fireDate
	  interval: (of_time_interval_t)interval
	    target: (id)target
	  selector: (SEL)selector
	    object: (id)object
	   repeats: (bool)repeats
{
	return [self OF_initWithFireDate: fireDate
				interval: interval
				  target: target
				selector: selector
				  object: object
				  object: nil
			       arguments: 1
				 repeats: repeats];
}

- initWithFireDate: (OFDate*)fireDate
	  interval: (of_time_interval_t)interval
	    target: (id)target
	  selector: (SEL)selector
	    object: (id)object1
	    object: (id)object2
	   repeats: (bool)repeats
{
	return [self OF_initWithFireDate: fireDate
				interval: interval
				  target: target
				selector: selector
				  object: object1
				  object: object2
			       arguments: 2
				 repeats: repeats];
}

#ifdef OF_HAVE_BLOCKS
- initWithFireDate: (OFDate*)fireDate
	   interval: (of_time_interval_t)interval
	    repeats: (bool)repeats
	      block: (of_timer_block_t)block
{
	self = [super init];

	@try {
		_fireDate = [fireDate retain];
		_interval = interval;
		_repeats = repeats;
		_block = [block copy];
		_valid = true;
# ifdef OF_HAVE_THREADS
		_condition = [[OFCondition alloc] init];
# endif
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}
#endif

- (void)dealloc
{
	/*
	 * The run loop references the timer, so it should never be deallocated
	 * if it is still in a run loop.
	 */
	assert(_inRunLoop == nil);

	[_fireDate release];
	[_target release];
	[_object1 release];
	[_object2 release];
#ifdef OF_HAVE_BLOCKS
	[_block release];
#endif
#ifdef OF_HAVE_THREADS
	[_condition release];
#endif

	[super dealloc];
}

- (of_comparison_result_t)compare: (id <OFComparing>)object
{
	OFTimer *timer;

	if (![object isKindOfClass: [OFTimer class]])
		@throw [OFInvalidArgumentException exception];

	timer = (OFTimer*)object;

	return [_fireDate compare: timer->_fireDate];
}

- (void)fire
{
	void *pool = objc_autoreleasePoolPush();
	id target = [[_target retain] autorelease];
	id object1 = [[_object1 retain] autorelease];
	id object2 = [[_object2 retain] autorelease];

	OF_ENSURE(_arguments <= 2);

	if (_repeats && _valid) {
		OFDate *old = _fireDate;
		_fireDate = [[OFDate alloc]
		    initWithTimeIntervalSinceNow: _interval];
		[old release];

		[[OFRunLoop currentRunLoop] addTimer: self];
	} else
		[self invalidate];

#ifdef OF_HAVE_BLOCKS
	if (_block != NULL)
		_block(self);
	else {
#endif
		switch (_arguments) {
		case 0:
			[target performSelector: _selector];
			break;
		case 1:
			[target performSelector: _selector
				     withObject: object1];
			break;
		case 2:
			[target performSelector: _selector
				     withObject: object1
				     withObject: object2];
			break;
		}
#ifdef OF_HAVE_BLOCKS
	}
#endif

#ifdef OF_HAVE_THREADS
	[_condition lock];
	@try {
		_done = true;
		[_condition signal];
	} @finally {
		[_condition unlock];
	}
#endif

	objc_autoreleasePoolPop(pool);
}

- (OFDate*)fireDate
{
	OF_GETTER(_fireDate, true)
}

- (void)setFireDate: (OFDate*)fireDate
{
	[self retain];
	@try {
		@synchronized (self) {
			[_inRunLoop OF_removeTimer: self];

			OF_SETTER(_fireDate, fireDate, true, 0)

			[_inRunLoop addTimer: self];
		}
	} @finally {
		[self release];
	}
}

- (of_time_interval_t)timeInterval
{
	return _interval;
}

- (void)invalidate
{
	_valid = false;

	[_target release];
	[_object1 release];
	[_object2 release];
	_target = nil;
	_object1 = nil;
	_object2 = nil;
}

- (bool)isValid
{
	return _valid;
}

#ifdef OF_HAVE_THREADS
- (void)waitUntilDone
{
	[_condition lock];
	@try {
		if (_done) {
			_done = false;
			return;
		}

		[_condition wait];
	} @finally {
		[_condition unlock];
	}
}
#endif

- (void)OF_setInRunLoop: (OFRunLoop*)inRunLoop
{
	OF_SETTER(_inRunLoop, inRunLoop, true, 0)
}
@end
