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

#import "OFObject.h"
#import "OFEnumerator.h"

OF_ASSUME_NONNULL_BEGIN

/*! @file */

/*!
 * @struct of_map_table_functions_t OFMapTable.h ObjFW/OFMapTable.h
 *
 * @brief A struct describing the functions to be used by the map table.
 */
typedef struct {
	/// The function to retain keys / values
	void *OF_NONNULL (*OF_NULLABLE retain)(void *value);
	/// The function to release keys / values
	void (*OF_NULLABLE release)(void *value);
	/// The function to hash keys
	uint32_t (*OF_NULLABLE hash)(void *value);
	/// The function to compare keys / values
	bool (*OF_NULLABLE equal)(void *value1, void *value2);
} of_map_table_functions_t;

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief A block for enumerating an OFMapTable.
 *
 * @param key The current key
 * @param value The current value
 * @param stop A pointer to a variable that can be set to true to stop the
 *	       enumeration
 */
typedef void (^of_map_table_enumeration_block_t)(void *key, void *value,
    bool *stop);

/*!
 * @brief A block for replacing values in an OFMapTable.
 *
 * @param key The key of the value to replace
 * @param value The value to replace
 * @return The value to replace the value with
 */
typedef void *OF_NONNULL (^of_map_table_replace_block_t)(
    void *key, void *value);
#endif

@class OFMapTableEnumerator;

/*!
 * @class OFMapTable OFMapTable.h ObjFW/OFMapTable.h
 *
 * @brief A class similar to OFDictionary, but providing more options how keys
 *	  and values should be retained, released, compared and hashed.
 */
@interface OFMapTable: OFObject <OFCopying, OFFastEnumeration>
{
	of_map_table_functions_t _keyFunctions, _valueFunctions;
	struct of_map_table_bucket **_buckets;
	uint32_t _count, _capacity;
	uint8_t _rotate;
	unsigned long _mutations;
}

/*!
 * @brief Creates a new OFMapTable with the specified key and value functions.
 *
 * @param keyFunctions A structure of functions for handling keys
 * @param valueFunctions A structure of functions for handling values
 * @return A new autoreleased OFMapTable
 */
+ (instancetype)mapTableWithKeyFunctions: (of_map_table_functions_t)keyFunctions
			  valueFunctions: (of_map_table_functions_t)
					      valueFunctions;

/*!
 * @brief Creates a new OFMapTable with the specified key functions, value
 *	  functions and capacity.
 *
 * @param keyFunctions A structure of functions for handling keys
 * @param valueFunctions A structure of functions for handling values
 * @param capacity A hint about the count of elements expected to be in the map
 *	  table
 * @return A new autoreleased OFMapTable
 */
+ (instancetype)mapTableWithKeyFunctions: (of_map_table_functions_t)keyFunctions
			  valueFunctions: (of_map_table_functions_t)
					      valueFunctions
				capacity: (size_t)capacity;

/*!
 * @brief Initializes an already allocated OFMapTable with the specified key
 *	  and value functions.
 *
 * @param keyFunctions A structure of functions for handling keys
 * @param valueFunctions A structure of functions for handling values
 * @return An initialized OFMapTable
 */
- initWithKeyFunctions: (of_map_table_functions_t)keyFunctions
	valueFunctions: (of_map_table_functions_t)valueFunctions;

/*!
 * @brief Initializes an already allocated OFMapTable with the specified key
 *	  functions, value functions and capacity.
 *
 * @param keyFunctions A structure of functions for handling keys
 * @param valueFunctions A structure of functions for handling values
 * @param capacity A hint about the count of elements expected to be in the map
 *	  table
 * @return An initialized OFMapTable
 */
- initWithKeyFunctions: (of_map_table_functions_t)keyFunctions
	valueFunctions: (of_map_table_functions_t)valueFunctions
	      capacity: (size_t)capacity;

/*!
 * @brief Returns the number of objects in the map table.
 *
 * @return The number of objects in the map table
 */
- (size_t)count;

/*!
 * @brief Returns the value for the given key or NULL if the key was not found.
 *
 * @param key The key whose object should be returned
 * @return The value for the given key or NULL if the key was not found
 */
- (nullable void*)valueForKey: (void*)key;

/*!
 * @brief Sets a value for a key.
 *
 * @param key The key to set
 * @param value The value to set the key to
 */
- (void)setValue: (void*)value
	  forKey: (void*)key;

/*!
 * @brief Removes the value for the specified key from the map table.
 *
 * @param key The key whose object should be removed
 */
- (void)removeValueForKey: (void*)key;

/*!
 * @brief Removes all values.
 */
- (void)removeAllValues;

/*!
 * @brief Checks whether the map table contains a value equal to the specified
 *	  value.
 *
 * @param value The value which is checked for being in the map table
 * @return A boolean whether the map table contains the specified value
*/
- (bool)containsValue: (nullable void*)value;

/*!
 * @brief Checks whether the map table contains a value with the specified
 *        address.
 *
 * @param value The value which is checked for being in the map table
 * @return A boolean whether the map table contains a value with the specified
 *	   address.
 */
- (bool)containsValueIdenticalTo: (nullable void*)value;

/*!
 * @brief Returns an OFMapTableEnumerator to enumerate through the map table's
 *	  keys.
 *
 * @return An OFMapTableEnumerator to enumerate through the map table's keys
 */
- (OFMapTableEnumerator*)keyEnumerator;

/*!
 * @brief Returns an OFMapTableEnumerator to enumerate through the map table's
 *	  values.
 *
 * @return An OFMapTableEnumerator to enumerate through the map table's values
 */
- (OFMapTableEnumerator*)valueEnumerator;

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief Executes a block for each key / object pair.
 *
 * @param block The block to execute for each key / object pair.
 */
- (void)enumerateKeysAndValuesUsingBlock:
    (of_map_table_enumeration_block_t)block;

/*!
 * @brief Replaces each value with the value returned by the block.
 *
 * @param block The block which returns a new value for each value
 */
- (void)replaceValuesUsingBlock: (of_map_table_replace_block_t)block;
#endif

/*!
 * @brief Returns the key functions used by the map table.
 *
 * @return The key functions used by the map table
 */
- (of_map_table_functions_t)keyFunctions;

/*!
 * @brief Returns the value functions used by the map table.
 *
 * @return The value functions used by the map table
 */
- (of_map_table_functions_t)valueFunctions;
@end

/*!
 * @class OFMapTableEnumerator OFMapTable.h ObjFW/OFMapTable.h
 *
 * @brief A class which provides methods to enumerate through an OFMapTable's
 *	  keys or values.
 */
@interface OFMapTableEnumerator: OFObject
{
	OFMapTable *_mapTable;
	struct of_map_table_bucket **_buckets;
	uint32_t _capacity;
	unsigned long _mutations;
	unsigned long *_mutationsPtr;
	uint32_t _position;
}

/*!
 * @brief Returns the next value.
 *
 * @return The next value
 */
- (void*)nextValue;

/*!
 * @brief Resets the enumerator, so the next call to nextKey returns the first
 *	  key again.
 */
- (void)reset;
@end

OF_ASSUME_NONNULL_END
