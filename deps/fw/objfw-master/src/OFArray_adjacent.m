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

#include <stdarg.h>

#import "OFArray_adjacent.h"
#import "OFMutableArray_adjacent.h"
#import "OFArray_adjacentSubarray.h"
#import "OFDataArray.h"
#import "OFString.h"
#import "OFXMLElement.h"

#import "OFEnumerationMutationException.h"
#import "OFInvalidArgumentException.h"
#import "OFOutOfRangeException.h"

@implementation OFArray_adjacent
- init
{
	self = [super init];

	@try {
		_array = [[OFDataArray alloc] initWithItemSize: sizeof(id)];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithObject: (id)object
{
	self = [self init];

	@try {
		if (object == nil)
			@throw [OFInvalidArgumentException exception];

		[_array addItem: &object];
		[object retain];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithObject: (id)firstObject
       arguments: (va_list)arguments
{
	self = [self init];

	@try {
		id object;

		[_array addItem: &firstObject];
		[firstObject retain];

		while ((object = va_arg(arguments, id)) != nil) {
			[_array addItem: &object];
			[object retain];
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithArray: (OFArray*)array
{
	id const *objects;
	size_t i, count;

	self = [super init];

	if (array == nil)
		return self;

	@try {
		objects = [array objects];
		count = [array count];

		_array = [[OFDataArray alloc] initWithItemSize: sizeof(id)
						      capacity: count];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	@try {
		for (i = 0; i < count; i++)
			[objects[i] retain];

		[_array addItems: objects
			   count: count];
	} @catch (id e) {
		for (i = 0; i < count; i++)
			[objects[i] release];

		/* Prevent double-release of objects */
		[_array release];
		_array = nil;

		[self release];
		@throw e;
	}

	return self;
}

- initWithObjects: (id const*)objects
	    count: (size_t)count
{
	self = [self init];

	@try {
		size_t i;
		bool ok = true;

		for (i = 0; i < count; i++) {
			if (objects[i] == nil)
				ok = false;

			[objects[i] retain];
		}

		if (!ok)
			@throw [OFInvalidArgumentException exception];

		[_array addItems: objects
			   count: count];
	} @catch (id e) {
		size_t i;

		for (i = 0; i < count; i++)
			[objects[i] release];

		[self release];
		@throw e;
	}

	return self;
}

- initWithSerialization: (OFXMLElement*)element
{
	self = [self init];

	@try {
		void *pool = objc_autoreleasePoolPush();
		OFEnumerator *enumerator;
		OFXMLElement *child;

		if ((![[element name] isEqual: @"OFArray"] &&
		    ![[element name] isEqual: @"OFMutableArray"]) ||
		    ![[element namespace] isEqual: OF_SERIALIZATION_NS])
			@throw [OFInvalidArgumentException exception];

		enumerator = [[element elementsForNamespace:
		    OF_SERIALIZATION_NS] objectEnumerator];

		while ((child = [enumerator nextObject]) != nil) {
			void *pool2 = objc_autoreleasePoolPush();
			id object;

			object = [child objectByDeserializing];
			[_array addItem: &object];
			[object retain];

			objc_autoreleasePoolPop(pool2);
		}

		objc_autoreleasePoolPop(pool);
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (size_t)count
{
	return [_array count];
}

- (id const*)objects
{
	return [_array items];
}

- (id)objectAtIndex: (size_t)index
{
	return *((id*)[_array itemAtIndex: index]);
}

- (id)objectAtIndexedSubscript: (size_t)index
{
	return *((id*)[_array itemAtIndex: index]);
}

- (void)getObjects: (id*)buffer
	   inRange: (of_range_t)range
{
	id *objects = [_array items];
	size_t i, count = [_array count];

	if (range.length > SIZE_MAX - range.location ||
	    range.location + range.length > count)
		@throw [OFOutOfRangeException exception];

	for (i = 0; i < range.length; i++)
		buffer[i] = objects[range.location + i];
}

- (size_t)indexOfObject: (id)object
{
	id *objects;
	size_t i, count;

	if (object == nil)
		return OF_NOT_FOUND;

	objects = [_array items];
	count = [_array count];

	for (i = 0; i < count; i++)
		if ([objects[i] isEqual: object])
			return i;

	return OF_NOT_FOUND;
}

- (size_t)indexOfObjectIdenticalTo: (id)object
{
	id *objects;
	size_t i, count;

	if (object == nil)
		return OF_NOT_FOUND;

	objects = [_array items];
	count = [_array count];

	for (i = 0; i < count; i++)
		if (objects[i] == object)
			return i;

	return OF_NOT_FOUND;
}


- (OFArray*)objectsInRange: (of_range_t)range
{
	if (range.length > SIZE_MAX - range.location ||
	    range.location + range.length > [_array count])
		@throw [OFOutOfRangeException exception];

	if ([self isKindOfClass: [OFMutableArray class]])
		return [OFArray
		    arrayWithObjects: (id*)[_array items] + range.location
			       count: range.length];

	return [OFArray_adjacentSubarray arrayWithArray: self
						  range: range];
}

- (bool)isEqual: (id)object
{
	OFArray *otherArray;
	id const *objects, *otherObjects;
	size_t i, count;

	if (![object isKindOfClass: [OFArray_adjacent class]] &&
	    ![object isKindOfClass: [OFMutableArray_adjacent class]])
		return [super isEqual: object];

	otherArray = object;

	count = [_array count];

	if (count != [otherArray count])
		return false;

	objects = [_array items];
	otherObjects = [otherArray objects];

	for (i = 0; i < count; i++)
		if (![objects[i] isEqual: otherObjects[i]])
			return false;

	return true;
}

- (uint32_t)hash
{
	id *objects = [_array items];
	size_t i, count = [_array count];
	uint32_t hash;

	OF_HASH_INIT(hash);

	for (i = 0; i < count; i++)
		OF_HASH_ADD_HASH(hash, [objects[i] hash]);

	OF_HASH_FINALIZE(hash);

	return hash;
}

- (int)countByEnumeratingWithState: (of_fast_enumeration_state_t*)state
			   objects: (id*)objects
			     count: (int)count_
{
	size_t count = [_array count];

	if (count > INT_MAX)
		/*
		 * Use the implementation from OFArray, which is slower, but can
		 * enumerate in chunks.
		 */
		return [super countByEnumeratingWithState: state
						  objects: objects
						    count: count_];

	if (state->state >= count)
		return 0;

	state->state = (unsigned long)count;
	state->itemsPtr = [_array items];
	state->mutationsPtr = (unsigned long*)self;

	return (int)count;
}

#ifdef OF_HAVE_BLOCKS
- (void)enumerateObjectsUsingBlock: (of_array_enumeration_block_t)block
{
	id *objects = [_array items];
	size_t i, count = [_array count];
	bool stop = false;

	for (i = 0; i < count && !stop; i++)
		block(objects[i], i, &stop);
}
#endif

- (void)dealloc
{
	id *objects = [_array items];
	size_t i, count = [_array count];

	for (i = 0; i < count; i++)
		[objects[i] release];

	[_array release];

	[super dealloc];
}
@end
