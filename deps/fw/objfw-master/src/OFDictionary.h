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

@class OFArray OF_GENERIC(ObjectType);

#ifdef OF_HAVE_BLOCKS
typedef void (^of_dictionary_enumeration_block_t)(id key, id object,
     bool *stop);
typedef bool (^of_dictionary_filter_block_t)(id key, id object);
typedef id OF_NONNULL (^of_dictionary_map_block_t)(id key, id object);
#endif

/*!
 * @class OFDictionary OFDictionary.h ObjFW/OFDictionary.h
 *
 * @brief An abstract class for storing objects in a dictionary.
 *
 * Keys are copied and thus must conform to the OFCopying protocol.
 *
 * @note Fast enumeration on a dictionary enumerates through the keys of the
 *	 dictionary.
 */
#ifdef OF_HAVE_GENERICS
@interface OFDictionary <KeyType, ObjectType>:
#else
# ifndef DOXYGEN
#  define KeyType id
#  define ObjectType id
# endif
@interface OFDictionary:
#endif
    OFObject <OFCopying, OFMutableCopying, OFCollection, OFSerialization,
    OFJSONRepresentation, OFMessagePackRepresentation>
/*!
 * @brief Creates a new OFDictionary.
 *
 * @return A new autoreleased OFDictionary
 */
+ (instancetype)dictionary;

/*!
 * @brief Creates a new OFDictionary with the specified dictionary.
 *
 * @param dictionary An OFDictionary
 * @return A new autoreleased OFDictionary
 */
+ (instancetype)dictionaryWithDictionary:
   (OFDictionary OF_GENERIC(KeyType, ObjectType)*)dictionary;

/*!
 * @brief Creates a new OFDictionary with the specified key and object.
 *
 * @param key The key
 * @param object The object
 * @return A new autoreleased OFDictionary
 */
+ (instancetype)dictionaryWithObject: (ObjectType)object
			      forKey: (KeyType)key;

/*!
 * @brief Creates a new OFDictionary with the specified keys and objects.
 *
 * @param keys An array of keys
 * @param objects An array of objects
 * @return A new autoreleased OFDictionary
 */
+ (instancetype)
    dictionaryWithObjects: (OFArray OF_GENERIC(ObjectType)*)objects
		  forKeys: (OFArray OF_GENERIC(KeyType)*)keys;

/*!
 * @brief Creates a new OFDictionary with the specified keys and objects.
 *
 * @param keys An array of keys
 * @param objects An array of objects
 * @param count The number of objects in the arrays
 * @return A new autoreleased OFDictionary
 */
+ (instancetype)
    dictionaryWithObjects: (ObjectType const OF_NONNULL *OF_NONNULL)objects
		  forKeys: (KeyType const OF_NONNULL *OF_NONNULL)keys
		    count: (size_t)count;

/*!
 * @brief Creates a new OFDictionary with the specified keys objects.
 *
 * @param firstKey The first key
 * @return A new autoreleased OFDictionary
 */
+ (instancetype)dictionaryWithKeysAndObjects: (KeyType)firstKey, ...
    OF_SENTINEL;

/*!
 * @brief Initializes an already allocated OFDictionary with the specified
 *	  OFDictionary.
 *
 * @param dictionary An OFDictionary
 * @return An initialized OFDictionary
 */
- initWithDictionary: (OFDictionary OF_GENERIC(KeyType, ObjectType)*)dictionary;

/*!
 * @brief Initializes an already allocated OFDictionary with the specified key
 *	  and object.
 *
 * @param key The key
 * @param object The object
 * @return An initialized OFDictionary
 */
- initWithObject: (ObjectType)object
	  forKey: (KeyType)key;

/*!
 * @brief Initializes an already allocated OFDictionary with the specified keys
 *	  and objects.
 *
 * @param keys An array of keys
 * @param objects An array of objects
 * @return An initialized OFDictionary
 */
- initWithObjects: (OFArray OF_GENERIC(ObjectType)*)objects
	  forKeys: (OFArray OF_GENERIC(KeyType)*)keys;

/*!
 * @brief Initializes an already allocated OFDictionary with the specified keys
 *	  and objects.
 *
 * @param keys An array of keys
 * @param objects An array of objects
 * @param count The number of objects in the arrays
 * @return An initialized OFDictionary
 */
- initWithObjects: (ObjectType const OF_NONNULL *OF_NONNULL)objects
	  forKeys: (KeyType const OF_NONNULL *OF_NONNULL)keys
	    count: (size_t)count;

/*!
 * @brief Initializes an already allocated OFDictionary with the specified keys
 *	  and objects.
 *
 * @param firstKey The first key
 * @return An initialized OFDictionary
 */
- initWithKeysAndObjects: (KeyType)firstKey, ... OF_SENTINEL;

/*!
 * @brief Initializes an already allocated OFDictionary with the specified key
 *	  and va_list.
 *
 * @param firstKey The first key
 * @param arguments A va_list of the other arguments
 * @return An initialized OFDictionary
 */
- initWithKey: (KeyType)firstKey
    arguments: (va_list)arguments;

/*!
 * @brief Returns the object for the given key or nil if the key was not found.
 *
 * @warning The returned object is *not* retained and autoreleased for
 *	    performance reasons!
 *
 * @param key The key whose object should be returned
 * @return The object for the given key or nil if the key was not found
 */
- (nullable ObjectType)objectForKey: (KeyType)key;
- (nullable ObjectType)objectForKeyedSubscript: (KeyType)key;

/*!
 * @brief Checks whether the dictionary contains an object equal to the
 *	  specified object.
 *
 * @param object The object which is checked for being in the dictionary
 * @return A boolean whether the dictionary contains the specified object
 */
- (bool)containsObject: (nullable ObjectType)object;

/*!
 * @brief Checks whether the dictionary contains an object with the specified
 *	  address.
 *
 * @param object The object which is checked for being in the dictionary
 * @return A boolean whether the dictionary contains an object with the
 *	   specified address
 */
- (bool)containsObjectIdenticalTo: (nullable ObjectType)object;

/*!
 * @brief Returns an array of all keys.
 *
 * @return An array of all keys
 */
- (OFArray OF_GENERIC(KeyType)*)allKeys;

/*!
 * @brief Returns an array of all objects.
 *
 * @return An array of all objects
 */
- (OFArray OF_GENERIC(ObjectType)*)allObjects;

/*!
 * @brief Returns an OFEnumerator to enumerate through the dictionary's keys.
 *
 * @return An OFEnumerator to enumerate through the dictionary's keys
 */
- (OFEnumerator OF_GENERIC(KeyType)*)keyEnumerator;

/*!
 * @brief Returns an OFEnumerator to enumerate through the dictionary's objects.
 *
 * @return An OFEnumerator to enumerate through the dictionary's objects
 */
- (OFEnumerator OF_GENERIC(ObjectType)*)objectEnumerator;

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief Executes a block for each key / object pair.
 *
 * @param block The block to execute for each key / object pair.
 */
- (void)enumerateKeysAndObjectsUsingBlock:
    (of_dictionary_enumeration_block_t)block;

/*!
 * @brief Creates a new dictionary, mapping each object using the specified
 *	  block.
 *
 * @param block A block which maps an object for each object
 * @return A new autoreleased OFDictionary
 */
- (OFDictionary OF_GENERIC(KeyType, id)*)mappedDictionaryUsingBlock:
    (of_dictionary_map_block_t)block;

/*!
 * @brief Creates a new dictionary, only containing the objects for which the
 *	  block returns true.
 *
 * @param block A block which determines if the object should be in the new
 *		dictionary
 * @return A new autoreleased OFDictionary
 */
- (OFDictionary OF_GENERIC(KeyType, ObjectType)*)filteredDictionaryUsingBlock:
    (of_dictionary_filter_block_t)block;
#endif
@end
#if !defined(OF_HAVE_GENERICS) && !defined(DOXYGEN)
# undef KeyType
# undef ObjectType
#endif

OF_ASSUME_NONNULL_END

#import "OFMutableDictionary.h"

#ifndef NSINTEGER_DEFINED
/* Required for dictionary literals to work */
@compatibility_alias NSDictionary OFDictionary;
#endif
