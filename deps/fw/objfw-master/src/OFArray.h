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

#ifndef __STDC_LIMIT_MACROS
# define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
# define __STDC_CONSTANT_MACROS
#endif

#include <stdarg.h>

#import "OFObject.h"
#import "OFCollection.h"
#import "OFEnumerator.h"
#import "OFSerialization.h"
#import "OFJSONRepresentation.h"
#import "OFMessagePackRepresentation.h"

OF_ASSUME_NONNULL_BEGIN

/*! @file */

@class OFString;

enum {
	OF_ARRAY_SKIP_EMPTY = 1,
	OF_ARRAY_SORT_DESCENDING = 2,
};

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief A block for enumerating an OFArray.
 *
 * @param object The current object
 * @param index The index of the current object
 * @param stop A pointer to a variable that can be set to true to stop the
 *	       enumeration
 */
typedef void (^of_array_enumeration_block_t)(id object, size_t index,
    bool *stop);

/*!
 * @brief A block for filtering an OFArray.
 *
 * @param object The object to inspect
 * @param index The index of the object to inspect
 * @return Whether the object should be in the filtered array
 */
typedef bool (^of_array_filter_block_t)(id object, size_t index);

/*!
 * @brief A block for mapping objects to objects in an OFArray.
 *
 * @param object The object to map
 * @param index The index of the object to map
 * @return The object to map to
 */
typedef id OF_NONNULL (^of_array_map_block_t)(id object, size_t index);

/*!
 * @brief A block for folding an OFArray.
 *
 * @param left The object to which the object has been folded so far
 * @param right The object that should be added to the left object
 * @return The left and right side folded into one object
 */
typedef id OF_NULLABLE (^of_array_fold_block_t)(id OF_NULLABLE left, id right);
#endif

/*!
 * @class OFArray OFArray.h ObjFW/OFArray.h
 *
 * @brief An abstract class for storing objects in an array.
 */
#ifdef OF_HAVE_GENERICS
@interface OFArray <ObjectType>:
#else
# ifndef DOXYGEN
#  define ObjectType id
# endif
@interface OFArray:
#endif
    OFObject <OFCopying, OFMutableCopying, OFCollection, OFSerialization,
    OFJSONRepresentation, OFMessagePackRepresentation>
#ifdef OF_HAVE_PROPERTIES
@property (readonly) size_t count;
#endif

/*!
 * @brief Creates a new OFArray.
 *
 * @return A new autoreleased OFArray
 */
+ (instancetype)array;

/*!
 * @brief Creates a new OFArray with the specified object.
 *
 * @param object An object
 * @return A new autoreleased OFArray
 */
+ (instancetype)arrayWithObject: (ObjectType)object;

/*!
 * @brief Creates a new OFArray with the specified objects, terminated by nil.
 *
 * @param firstObject The first object in the array
 * @return A new autoreleased OFArray
 */
+ (instancetype)arrayWithObjects: (ObjectType)firstObject, ... OF_SENTINEL;

/*!
 * @brief Creates a new OFArray with the objects from the specified array.
 *
 * @param array An array
 * @return A new autoreleased OFArray
 */
+ (instancetype)arrayWithArray: (OFArray OF_GENERIC(ObjectType)*)array;

/*!
 * @brief Creates a new OFArray with the objects from the specified C array of
 *	  the specified length.
 *
 * @param objects A C array of objects
 * @param count The length of the C array
 * @return A new autoreleased OFArray
 */
+ (instancetype)
    arrayWithObjects: (ObjectType const OF_NONNULL *OF_NONNULL)objects
	       count: (size_t)count;

/*!
 * @brief Initializes an OFArray with the specified object.
 *
 * @param object An object
 * @return An initialized OFArray
 */
- initWithObject: (ObjectType)object;

/*!
 * @brief Initializes an OFArray with the specified objects.
 *
 * @param firstObject The first object
 * @return An initialized OFArray
 */
- initWithObjects: (ObjectType)firstObject, ... OF_SENTINEL;

/*!
 * @brief Initializes an OFArray with the specified object and a va_list.
 *
 * @param firstObject The first object
 * @param arguments A va_list
 * @return An initialized OFArray
 */
- initWithObject: (ObjectType)firstObject
       arguments: (va_list)arguments;

/*!
 * @brief Initializes an OFArray with the objects from the specified array.
 *
 * @param array An array
 * @return An initialized OFArray
 */
- initWithArray: (OFArray OF_GENERIC(ObjectType)*)array;

/*!
 * @brief Initializes an OFArray with the objects from the specified C array of
 *	  the specified length.
 *
 * @param objects A C array of objects
 * @param count The length of the C array
 * @return An initialized OFArray
 */
- initWithObjects: (ObjectType const OF_NONNULL *OF_NONNULL)objects
	    count: (size_t)count;

/*!
 * @brief Returns the object at the specified index in the array.
 *
 * @warning The returned object is *not* retained and autoreleased for
 *	    performance reasons!
 *
 * @param index The index of the object to return
 * @return The object at the specified index in the array
 */
- (ObjectType)objectAtIndex: (size_t)index;
- (ObjectType)objectAtIndexedSubscript: (size_t)index;

/*!
 * @brief Copies the objects at the specified range to the specified buffer.
 *
 * @param buffer The buffer to copy the objects to
 * @param range The range to copy
 */
- (void)getObjects: (ObjectType __unsafe_unretained OF_NONNULL *OF_NONNULL)
			buffer
	   inRange: (of_range_t)range;

/*!
 * @brief Returns the objects of the array as a C array.
 *
 * @return The objects of the array as a C array
 */
- (ObjectType const __unsafe_unretained OF_NONNULL *OF_NONNULL)objects;

/*!
 * @brief Returns the index of the first object that is equivalent to the
 *	  specified object or `OF_NOT_FOUND` if it was not found.
 *
 * @param object The object whose index is returned
 * @return The index of the first object equivalent to the specified object
 *	   or `OF_NOT_FOUND` if it was not found
 */
- (size_t)indexOfObject: (ObjectType)object;

/*!
 * @brief Returns the index of the first object that has the same address as the
 *	  specified object or `OF_NOT_FOUND` if it was not found.
 *
 * @param object The object whose index is returned
 * @return The index of the first object that has the same aaddress as
 *	   the specified object or `OF_NOT_FOUND` if it was not found
 */
- (size_t)indexOfObjectIdenticalTo: (ObjectType)object;

/*!
 * @brief Checks whether the array contains an object equal to the specified
 *	  object.
 *
 * @param object The object which is checked for being in the array
 * @return A boolean whether the array contains the specified object
 */
- (bool)containsObject: (nullable ObjectType)object;

/*!
 * @brief Checks whether the array contains an object with the specified
 *	  address.
 *
 * @param object The object which is checked for being in the array
 * @return A boolean whether the array contains an object with the specified
 *	   address
 */
- (bool)containsObjectIdenticalTo: (nullable ObjectType)object;

/*!
 * @brief Returns the first object of the array or nil.
 *
 * @warning The returned object is *not* retained and autoreleased for
 *	    performance reasons!
 *
 * @return The first object of the array or nil
 */
- (nullable ObjectType)firstObject;

/*!
 * @brief Returns the last object of the array or nil.
 *
 * @warning The returned object is *not* retained and autoreleased for
 *	    performance reasons!
 *
 * @return The last object of the array or nil
 */
- (nullable ObjectType)lastObject;

/*!
 * @brief Returns the objects in the specified range as a new OFArray.
 *
 * @param range The range for the subarray
 * @return The subarray as a new autoreleased OFArray
 */
- (OFArray OF_GENERIC(ObjectType)*)objectsInRange: (of_range_t)range;

/*!
 * @brief Creates a string by joining all objects of the array.
 *
 * @param separator The string with which the objects should be joined
 * @return A string containing all objects joined by the separator
 */
- (OFString*)componentsJoinedByString: (OFString*)separator;

/*!
 * @brief Creates a string by joining all objects of the array.
 *
 * @param separator The string with which the objects should be joined
 * @param options Options according to which the objects should be joined.@n
 *		  Possible values are:
 *		  Value                 | Description
 *		  ----------------------|----------------------
 * 		  `OF_ARRAY_SKIP_EMPTY` | Skip empty components
 * @return A string containing all objects joined by the separator
 */
- (OFString*)componentsJoinedByString: (OFString*)separator
			      options: (int)options;

/*!
 * @brief Creates a string by calling the selector on all objects of the array
 *	  and joining the strings returned by calling the selector.
 *
 * @param separator The string with which the objects should be joined
 * @param selector The selector to perform on the objects
 * @return A string containing all objects joined by the separator
 */
- (OFString*)componentsJoinedByString: (OFString*)separator
			usingSelector: (SEL)selector;

/*!
 * @brief Creates a string by calling the selector on all objects of the array
 *	  and joining the strings returned by calling the selector.
 *
 * @param separator The string with which the objects should be joined
 * @param selector The selector to perform on the objects
 * @param options Options according to which the objects should be joined.@n
 *		  Possible values are:
 *		  Value                 | Description
 *		  ----------------------|----------------------
 * 		  `OF_ARRAY_SKIP_EMPTY` | Skip empty components
 * @return A string containing all objects joined by the separator
 */
- (OFString*)componentsJoinedByString: (OFString*)separator
			usingSelector: (SEL)selector
			      options: (int)options;

/*!
 * @brief Performs the specified selector on all objects in the array.
 *
 * @param selector The selector to perform on all objects in the array
 */
- (void)makeObjectsPerformSelector: (SEL)selector;

/*!
 * @brief Performs the specified selector on all objects in the array with the
 *	  specified object.
 *
 * @param selector The selector to perform on all objects in the array
 * @param object The object to perform the selector with on all objects in the
 *	      array
 */
- (void)makeObjectsPerformSelector: (SEL)selector
			withObject: (nullable id)object;

/*!
 * @brief Returns a sorted copy of the array.
 *
 * @return A sorted copy of the array
 */
- (OFArray OF_GENERIC(ObjectType)*)sortedArray;

/*!
 * @brief Returns a sorted copy of the array.
 *
 * @param options The options to use when sorting the array.@n
 *		  Possible values are:
 *		  Value                      | Description
 *		  ---------------------------|-------------------------
 *		  `OF_ARRAY_SORT_DESCENDING` | Sort in descending order
 * @return A sorted copy of the array
 */
- (OFArray OF_GENERIC(ObjectType)*)sortedArrayWithOptions: (int)options;

/*!
 * @brief Returns a copy of the array with the order reversed.
 *
 * @return A copy of the array with the order reversed
 */
- (OFArray OF_GENERIC(ObjectType)*)reversedArray;

/*!
 * @brief Creates a new array with the specified object added.
 *
 * @param object The object to add
 * @return A new array with the specified object added
 */
- (OFArray OF_GENERIC(ObjectType)*)arrayByAddingObject: (ObjectType)object;

/*!
 * @brief Creates a new array with the objects from the specified array added.
 *
 * @param array The array with objects to add
 * @return A new array with the objects from the specified array added
 */
- (OFArray OF_GENERIC(ObjectType)*)arrayByAddingObjectsFromArray:
    (OFArray OF_GENERIC(ObjectType)*)array;

/*!
 * @brief Creates a new array with the specified object removed.
 *
 * @param object The object to remove
 * @return A new array with the specified object removed
 */
- (OFArray OF_GENERIC(ObjectType)*)arrayByRemovingObject: (ObjectType)object;

/*!
 * @brief Returns an OFEnumerator to enumerate through all objects of the
 *	  array.
 *
 * @returns An OFEnumerator to enumerate through all objects of the array
 */
- (OFEnumerator OF_GENERIC(ObjectType)*)objectEnumerator;

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief Executes a block for each object.
 *
 * @param block The block to execute for each object
 */
- (void)enumerateObjectsUsingBlock: (of_array_enumeration_block_t)block;

/*!
 * @brief Creates a new array, mapping each object using the specified block.
 *
 * @param block A block which maps an object for each object
 * @return A new, autoreleased OFArray
 */
- (OFArray*)mappedArrayUsingBlock: (of_array_map_block_t)block;

/*!
 * @brief Creates a new array, only containing the objects for which the block
 *	  returns true.
 *
 * @param block A block which determines if the object should be in the new
 *		array
 * @return A new, autoreleased OFArray
 */
- (OFArray OF_GENERIC(ObjectType)*)filteredArrayUsingBlock:
    (of_array_filter_block_t)block;

/*!
 * @brief Folds the array to a single object using the specified block.
 *
 * If the array is empty, it will return nil.
 *
 * If there is only one object in the array, that object will be returned and
 * the block will not be invoked.
 *
 * If there are at least two objects, the block is invoked for each object
 * except the first, where left is always to what the array has already been
 * folded and right what should be added to left.
 *
 * @param block A block which folds two objects into one, which is called for
 *		all objects except the first
 * @return The array folded to a single object
 */
- (nullable id)foldUsingBlock: (of_array_fold_block_t)block;
#endif
@end
#if !defined(OF_HAVE_GENERICS) && !defined(DOXYGEN)
# undef ObjectType
#endif

@interface OFArrayEnumerator: OFEnumerator
{
	OFArray	      *_array;
	size_t	      _count;
	unsigned long _mutations;
	unsigned long *_mutationsPtr;
	size_t	      _position;
}

- initWithArray: (OFArray*)data
   mutationsPtr: (unsigned long *OF_NULLABLE)mutationsPtr;
@end

OF_ASSUME_NONNULL_END

#import "OFMutableArray.h"

#ifndef NSINTEGER_DEFINED
/* Required for array literals to work */
@compatibility_alias NSArray OFArray;
#endif
