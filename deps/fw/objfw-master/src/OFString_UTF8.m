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
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <sys/types.h>

#import "OFString_UTF8.h"
#import "OFString_UTF8+Private.h"
#import "OFMutableString_UTF8.h"
#import "OFArray.h"

#import "OFInitializationFailedException.h"
#import "OFInvalidArgumentException.h"
#import "OFInvalidEncodingException.h"
#import "OFInvalidFormatException.h"
#import "OFOutOfMemoryException.h"
#import "OFOutOfRangeException.h"

#import "of_asprintf.h"
#import "unicode.h"

extern const of_char16_t of_iso_8859_15[128];
extern const of_char16_t of_windows_1252[128];
extern const of_char16_t of_codepage_437[128];

static inline int
memcasecmp(const char *first, const char *second, size_t length)
{
	size_t i;

	for (i = 0; i < length; i++) {
		if (tolower((int)first[i]) > tolower((int)second[i]))
			return OF_ORDERED_DESCENDING;
		if (tolower((int)first[i]) < tolower((int)second[i]))
			return OF_ORDERED_ASCENDING;
	}

	return OF_ORDERED_SAME;
}

int
of_string_utf8_check(const char *UTF8String, size_t UTF8Length, size_t *length)
{
	size_t i, tmpLength = UTF8Length;
	int isUTF8 = 0;

	for (i = 0; i < UTF8Length; i++) {
		/* No sign of UTF-8 here */
		if OF_LIKELY (!(UTF8String[i] & 0x80))
			continue;

		isUTF8 = 1;

		/* We're missing a start byte here */
		if OF_UNLIKELY (!(UTF8String[i] & 0x40))
			return -1;

		/* 2 byte sequences for code points 0 - 127 are forbidden */
		if OF_UNLIKELY ((UTF8String[i] & 0x7E) == 0x40)
			return -1;

		/* We have at minimum a 2 byte character -> check next byte */
		if OF_UNLIKELY (UTF8Length <= i + 1 ||
		    (UTF8String[i + 1] & 0xC0) != 0x80)
			return -1;

		/* Check if we have at minimum a 3 byte character */
		if OF_LIKELY (!(UTF8String[i] & 0x20)) {
			i++;
			tmpLength--;
			continue;
		}

		/* We have at minimum a 3 byte char -> check second next byte */
		if OF_UNLIKELY (UTF8Length <= i + 2 ||
		    (UTF8String[i + 2] & 0xC0) != 0x80)
			return -1;

		/* Check if we have a 4 byte character */
		if OF_LIKELY (!(UTF8String[i] & 0x10)) {
			i += 2;
			tmpLength -= 2;
			continue;
		}

		/* We have a 4 byte character -> check third next byte */
		if OF_UNLIKELY (UTF8Length <= i + 3 ||
		    (UTF8String[i + 3] & 0xC0) != 0x80)
			return -1;

		/*
		 * Just in case, check if there's a 5th character, which is
		 * forbidden by UTF-8
		 */
		if OF_UNLIKELY (UTF8String[i] & 0x08)
			return -1;

		i += 3;
		tmpLength -= 3;
	}

	if (length != NULL)
		*length = tmpLength;

	return isUTF8;
}

size_t
of_string_utf8_get_index(const char *string, size_t position)
{
	size_t i, index = position;

	for (i = 0; i < position; i++)
		if OF_UNLIKELY ((string[i] & 0xC0) == 0x80)
			index--;

	return index;
}

size_t
of_string_utf8_get_position(const char *string, size_t index, size_t length)
{
	size_t i;

	for (i = 0; i <= index; i++)
		if OF_UNLIKELY ((string[i] & 0xC0) == 0x80)
			if (++index > length)
				return OF_NOT_FOUND;

	return index;
}

@implementation OFString_UTF8
- init
{
	self = [super init];

	@try {
		_s = &_storage;

		_s->cString = [self allocMemoryWithSize: 1];
		_s->cString[0] = '\0';
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (instancetype)OF_initWithUTF8String: (const char*)UTF8String
			       length: (size_t)UTF8StringLength
			      storage: (char*)storage
{
	self = [super init];

	@try {
		if (UTF8StringLength >= 3 &&
		    memcmp(UTF8String, "\xEF\xBB\xBF", 3) == 0) {
			UTF8String += 3;
			UTF8StringLength -= 3;
		}

		_s = &_storage;

		_s->cString = storage;
		_s->cStringLength = UTF8StringLength;

		switch (of_string_utf8_check(UTF8String, UTF8StringLength,
		    &_s->length)) {
		case 1:
			_s->isUTF8 = true;
			break;
		case -1:
			@throw [OFInvalidEncodingException exception];
		}

		memcpy(_s->cString, UTF8String, UTF8StringLength);
		_s->cString[UTF8StringLength] = 0;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithCString: (const char*)cString
	 encoding: (of_string_encoding_t)encoding
	   length: (size_t)cStringLength
{
	self = [super init];

	@try {
		size_t i, j;
		const of_char16_t *table;

		if (encoding == OF_STRING_ENCODING_UTF_8 &&
		    cStringLength >= 3 &&
		    memcmp(cString, "\xEF\xBB\xBF", 3) == 0) {
			cString += 3;
			cStringLength -= 3;
		}

		_s = &_storage;

		_s->cString = [self allocMemoryWithSize: cStringLength + 1];
		_s->cStringLength = cStringLength;

		if (encoding == OF_STRING_ENCODING_UTF_8 ||
		    encoding == OF_STRING_ENCODING_ASCII) {
			switch (of_string_utf8_check(cString, cStringLength,
			    &_s->length)) {
			case 1:
				if (encoding == OF_STRING_ENCODING_ASCII)
					@throw [OFInvalidEncodingException
					    exception];

				_s->isUTF8 = true;
				break;
			case -1:
				@throw [OFInvalidEncodingException exception];
			}

			memcpy(_s->cString, cString, cStringLength);
			_s->cString[cStringLength] = 0;

			return self;
		}

		/* All other encodings we support are single byte encodings */
		_s->length = cStringLength;

		if (encoding == OF_STRING_ENCODING_ISO_8859_1) {
			for (i = j = 0; i < cStringLength; i++) {
				char buffer[4];
				size_t bytes;

				if (!(cString[i] & 0x80)) {
					_s->cString[j++] = cString[i];
					continue;
				}

				_s->isUTF8 = true;
				bytes = of_string_utf8_encode(
				    (uint8_t)cString[i], buffer);

				if (bytes == 0)
					@throw [OFInvalidEncodingException
					    exception];

				_s->cStringLength += bytes - 1;
				_s->cString = [self
				    resizeMemory: _s->cString
					    size: _s->cStringLength + 1];

				memcpy(_s->cString + j, buffer, bytes);
				j += bytes;
			}

			_s->cString[_s->cStringLength] = 0;

			return self;
		}

		switch (encoding) {
		case OF_STRING_ENCODING_ISO_8859_15:
			table = of_iso_8859_15;
			break;
		case OF_STRING_ENCODING_WINDOWS_1252:
			table = of_windows_1252;
			break;
		case OF_STRING_ENCODING_CODEPAGE_437:
			table = of_codepage_437;
			break;
		default:
			@throw [OFInvalidEncodingException exception];
		}

		for (i = j = 0; i < cStringLength; i++) {
			char buffer[4];
			of_unichar_t character;
			size_t characterBytes;

			if (!(cString[i] & 0x80)) {
				_s->cString[j++] = cString[i];
				continue;
			}

			character = table[(uint8_t)cString[i] - 128];

			if (character == 0xFFFD)
				@throw [OFInvalidEncodingException exception];

			_s->isUTF8 = true;
			characterBytes = of_string_utf8_encode(character,
			    buffer);

			if (characterBytes == 0)
				@throw [OFInvalidEncodingException exception];

			_s->cStringLength += characterBytes - 1;
			_s->cString = [self
			    resizeMemory: _s->cString
				    size: _s->cStringLength + 1];

			memcpy(_s->cString + j, buffer, characterBytes);
			j += characterBytes;
		}

		_s->cString[_s->cStringLength] = 0;
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithUTF8StringNoCopy: (char*)UTF8String
	      freeWhenDone: (bool)freeWhenDone
{
	self = [super init];

	@try {
		size_t UTF8StringLength = strlen(UTF8String);

		if (UTF8StringLength >= 3 &&
		    memcmp(UTF8String, "\xEF\xBB\xBF", 3) == 0) {
			UTF8String += 3;
			UTF8StringLength -= 3;
		}

		_s = &_storage;

		_s->cString = (char*)UTF8String;
		_s->cStringLength = UTF8StringLength;

		if (freeWhenDone)
			_s->freeWhenDone = UTF8String;

		switch (of_string_utf8_check(UTF8String, UTF8StringLength,
		    &_s->length)) {
		case 1:
			_s->isUTF8 = true;
			break;
		case -1:
			@throw [OFInvalidEncodingException exception];
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithString: (OFString*)string
{
	self = [super init];

	@try {
		_s = &_storage;

		_s->cStringLength = [string UTF8StringLength];

		if ([string isKindOfClass: [OFString_UTF8 class]] ||
		    [string isKindOfClass: [OFMutableString_UTF8 class]])
			_s->isUTF8 = ((OFString_UTF8*)string)->_s->isUTF8;
		else
			_s->isUTF8 = true;

		_s->length = [string length];

		_s->cString = [self allocMemoryWithSize: _s->cStringLength + 1];
		memcpy(_s->cString, [string UTF8String], _s->cStringLength + 1);
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithCharacters: (const of_unichar_t*)characters
	      length: (size_t)length
{
	self = [super init];

	@try {
		size_t i, j = 0;

		_s = &_storage;

		_s->cString = [self allocMemoryWithSize: (length * 4) + 1];
		_s->length = length;

		for (i = 0; i < length; i++) {
			size_t len = of_string_utf8_encode(characters[i],
			    _s->cString + j);

			if (len == 0)
				@throw [OFInvalidEncodingException exception];

			if (len > 1)
				_s->isUTF8 = true;

			j += len;
		}

		_s->cString[j] = '\0';
		_s->cStringLength = j;

		@try {
			_s->cString = [self resizeMemory: _s->cString
						    size: j + 1];
		} @catch (OFOutOfMemoryException *e) {
			/* We don't care, as we only tried to make it smaller */
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithUTF16String: (const of_char16_t*)string
	       length: (size_t)length
	    byteOrder: (of_byte_order_t)byteOrder
{
	self = [super init];

	@try {
		size_t i, j = 0;
		bool swap = false;

		if (length > 0 && *string == 0xFEFF) {
			string++;
			length--;
		} else if (length > 0 && *string == 0xFFFE) {
			swap = true;
			string++;
			length--;
		} else if (byteOrder != OF_BYTE_ORDER_NATIVE)
			swap = true;

		_s = &_storage;

		_s->cString = [self allocMemoryWithSize: (length * 4) + 1];
		_s->length = length;

		for (i = 0; i < length; i++) {
			of_unichar_t character =
			    (swap ? OF_BSWAP16(string[i]) : string[i]);
			size_t len;

			/* Missing high surrogate */
			if ((character & 0xFC00) == 0xDC00)
				@throw [OFInvalidEncodingException exception];

			if ((character & 0xFC00) == 0xD800) {
				of_char16_t nextCharacter;

				if (length <= i + 1)
					@throw [OFInvalidEncodingException
					    exception];

				nextCharacter = (swap
				    ? OF_BSWAP16(string[i + 1])
				    : string[i + 1]);

				if ((nextCharacter & 0xFC00) != 0xDC00)
					@throw [OFInvalidEncodingException
					    exception];

				character = (((character & 0x3FF) << 10) |
				    (nextCharacter & 0x3FF)) + 0x10000;

				i++;
				_s->length--;
			}

			len = of_string_utf8_encode(character, _s->cString + j);

			if (len == 0)
				@throw [OFInvalidEncodingException exception];

			if (len > 1)
				_s->isUTF8 = true;

			j += len;
		}

		_s->cString[j] = '\0';
		_s->cStringLength = j;

		@try {
			_s->cString = [self resizeMemory: _s->cString
						    size: j + 1];
		} @catch (OFOutOfMemoryException *e) {
			/* We don't care, as we only tried to make it smaller */
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithUTF32String: (const of_char32_t*)characters
	       length: (size_t)length
	    byteOrder: (of_byte_order_t)byteOrder
{
	self = [super init];

	@try {
		size_t i, j = 0;
		bool swap = false;

		if (length > 0 && *characters == 0xFEFF) {
			characters++;
			length--;
		} else if (length > 0 && *characters == 0xFFFE0000) {
			swap = true;
			characters++;
			length--;
		} else if (byteOrder != OF_BYTE_ORDER_NATIVE)
			swap = true;

		_s = &_storage;

		_s->cString = [self allocMemoryWithSize: (length * 4) + 1];
		_s->length = length;

		for (i = 0; i < length; i++) {
			char buffer[4];
			size_t len = of_string_utf8_encode(
			    (swap ? OF_BSWAP32(characters[i]) : characters[i]),
			    buffer);

			switch (len) {
			case 1:
				_s->cString[j++] = buffer[0];
				break;
			case 2:
			case 3:
			case 4:
				_s->isUTF8 = true;

				memcpy(_s->cString + j, buffer, len);
				j += len;

				break;
			default:
				@throw [OFInvalidEncodingException exception];
			}
		}

		_s->cString[j] = '\0';
		_s->cStringLength = j;

		@try {
			_s->cString = [self resizeMemory: _s->cString
						    size: j + 1];
		} @catch (OFOutOfMemoryException *e) {
			/* We don't care, as we only tried to make it smaller */
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithFormat: (OFConstantString*)format
       arguments: (va_list)arguments
{
	self = [super init];

	@try {
		char *tmp;
		int cStringLength;

		if (format == nil)
			@throw [OFInvalidArgumentException exception];

		_s = &_storage;

		if ((cStringLength = of_vasprintf(&tmp, [format UTF8String],
		    arguments)) == -1)
			@throw [OFInvalidFormatException exception];

		_s->cStringLength = cStringLength;

		@try {
			switch (of_string_utf8_check(tmp, cStringLength,
			    &_s->length)) {
			case 1:
				_s->isUTF8 = true;
				break;
			case -1:
				@throw [OFInvalidEncodingException exception];
			}

			_s->cString = [self
			    allocMemoryWithSize: cStringLength + 1];
			memcpy(_s->cString, tmp, cStringLength + 1);
		} @finally {
			free(tmp);
		}
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- (void)dealloc
{
	if (_s != NULL && _s->freeWhenDone != NULL)
		free(_s->freeWhenDone);

	[super dealloc];
}

- (size_t)getCString: (char*)cString
	   maxLength: (size_t)maxLength
	    encoding: (of_string_encoding_t)encoding
{
	switch (encoding) {
	case OF_STRING_ENCODING_ASCII:
		if (_s->isUTF8)
			@throw [OFInvalidEncodingException exception];
		/* intentional fall-through */
	case OF_STRING_ENCODING_UTF_8:
		if (_s->cStringLength + 1 > maxLength)
			@throw [OFOutOfRangeException exception];

		memcpy(cString, _s->cString, _s->cStringLength + 1);

		return _s->cStringLength;
	default:
		return [super getCString: cString
			       maxLength: maxLength
				encoding: encoding];
	}
}

- (const char*)cStringWithEncoding: (of_string_encoding_t)encoding
{
	switch (encoding) {
	case OF_STRING_ENCODING_ASCII:
		if (_s->isUTF8)
			@throw [OFInvalidEncodingException exception];
		/* intentional fall-through */
	case OF_STRING_ENCODING_UTF_8:
		return _s->cString;
	default:
		return [super cStringWithEncoding: encoding];
	}
}

- (const char*)UTF8String
{
	return _s->cString;
}

- (size_t)length
{
	return _s->length;
}

- (size_t)cStringLengthWithEncoding: (of_string_encoding_t)encoding
{
	switch (encoding) {
	case OF_STRING_ENCODING_UTF_8:
	case OF_STRING_ENCODING_ASCII:
		return _s->cStringLength;
	default:
		return [super cStringLengthWithEncoding: encoding];
	}
}

- (size_t)UTF8StringLength
{
	return _s->cStringLength;
}

- (bool)isEqual: (id)object
{
	OFString_UTF8 *otherString;

	if (object == self)
		return true;

	if (![object isKindOfClass: [OFString class]])
		return false;

	otherString = object;

	if ([otherString UTF8StringLength] != _s->cStringLength ||
	    [otherString length] != _s->length)
		return false;

	if (([otherString isKindOfClass: [OFString_UTF8 class]] ||
	    [otherString isKindOfClass: [OFMutableString_UTF8 class]]) &&
	    _s->hashed && otherString->_s->hashed &&
	    _s->hash != otherString->_s->hash)
		return false;

	if (strcmp(_s->cString, [otherString UTF8String]) != 0)
		return false;

	return true;
}

- (of_comparison_result_t)compare: (id <OFComparing>)object
{
	OFString *otherString;
	size_t otherCStringLength, minimumCStringLength;
	int compare;

	if (object == self)
		return OF_ORDERED_SAME;

	if (![object isKindOfClass: [OFString class]])
		@throw [OFInvalidArgumentException exception];

	otherString = (OFString*)object;
	otherCStringLength = [otherString UTF8StringLength];
	minimumCStringLength = (_s->cStringLength > otherCStringLength
	    ? otherCStringLength : _s->cStringLength);

	if ((compare = memcmp(_s->cString, [otherString UTF8String],
	    minimumCStringLength)) == 0) {
		if (_s->cStringLength > otherCStringLength)
			return OF_ORDERED_DESCENDING;
		if (_s->cStringLength < otherCStringLength)
			return OF_ORDERED_ASCENDING;
		return OF_ORDERED_SAME;
	}

	if (compare > 0)
		return OF_ORDERED_DESCENDING;
	else
		return OF_ORDERED_ASCENDING;
}

- (of_comparison_result_t)caseInsensitiveCompare: (OFString*)otherString
{
	const char *otherCString;
	size_t i, j, otherCStringLength, minimumCStringLength;
	int compare;

	if (otherString == self)
		return OF_ORDERED_SAME;

	if (![otherString isKindOfClass: [OFString class]])
		@throw [OFInvalidArgumentException exception];

	otherCString = [otherString UTF8String];
	otherCStringLength = [otherString UTF8StringLength];

	if (!_s->isUTF8) {
		minimumCStringLength = (_s->cStringLength > otherCStringLength
		    ? otherCStringLength : _s->cStringLength);

		if ((compare = memcasecmp(_s->cString, otherCString,
		    minimumCStringLength)) == 0) {
			if (_s->cStringLength > otherCStringLength)
				return OF_ORDERED_DESCENDING;
			if (_s->cStringLength < otherCStringLength)
				return OF_ORDERED_ASCENDING;
			return OF_ORDERED_SAME;
		}

		if (compare > 0)
			return OF_ORDERED_DESCENDING;
		else
			return OF_ORDERED_ASCENDING;
	}

	i = j = 0;

	while (i < _s->cStringLength && j < otherCStringLength) {
		of_unichar_t c1, c2;
		size_t l1, l2;

		l1 = of_string_utf8_decode(_s->cString + i,
		    _s->cStringLength - i, &c1);
		l2 = of_string_utf8_decode(otherCString + j,
		    otherCStringLength - j, &c2);

		if (l1 == 0 || l2 == 0 || c1 > 0x10FFFF || c2 > 0x10FFFF)
			@throw [OFInvalidEncodingException exception];

		if (c1 >> 8 < OF_UNICODE_CASEFOLDING_TABLE_SIZE) {
			of_unichar_t tc =
			    of_unicode_casefolding_table[c1 >> 8][c1 & 0xFF];

			if (tc)
				c1 = tc;
		}

		if (c2 >> 8 < OF_UNICODE_CASEFOLDING_TABLE_SIZE) {
			of_unichar_t tc =
			    of_unicode_casefolding_table[c2 >> 8][c2 & 0xFF];

			if (tc)
				c2 = tc;
		}

		if (c1 > c2)
			return OF_ORDERED_DESCENDING;
		if (c1 < c2)
			return OF_ORDERED_ASCENDING;

		i += l1;
		j += l2;
	}

	if (_s->cStringLength - i > otherCStringLength - j)
		return OF_ORDERED_DESCENDING;
	else if (_s->cStringLength - i < otherCStringLength - j)
		return OF_ORDERED_ASCENDING;

	return OF_ORDERED_SAME;
}

- (uint32_t)hash
{
	size_t i;
	uint32_t hash;

	if (_s->hashed)
		return _s->hash;

	OF_HASH_INIT(hash);

	for (i = 0; i < _s->cStringLength; i++) {
		of_unichar_t c;
		size_t length;

		if ((length = of_string_utf8_decode(_s->cString + i,
		    _s->cStringLength - i, &c)) == 0)
			@throw [OFInvalidEncodingException exception];

		OF_HASH_ADD(hash, (c & 0xFF0000) >> 16);
		OF_HASH_ADD(hash, (c & 0x00FF00) >>  8);
		OF_HASH_ADD(hash,  c & 0x0000FF);

		i += length - 1;
	}

	OF_HASH_FINALIZE(hash);

	_s->hash = hash;
	_s->hashed = true;

	return hash;
}

- (of_unichar_t)characterAtIndex: (size_t)index
{
	of_unichar_t character;

	if (index >= _s->length)
		@throw [OFOutOfRangeException exception];

	if (!_s->isUTF8)
		return _s->cString[index];

	index = of_string_utf8_get_position(_s->cString, index,
	    _s->cStringLength);

	if (!of_string_utf8_decode(_s->cString + index,
	    _s->cStringLength - index, &character))
		@throw [OFInvalidEncodingException exception];

	return character;
}

- (void)getCharacters: (of_unichar_t*)buffer
	      inRange: (of_range_t)range
{
	/* TODO: Could be slightly optimized */
	void *pool = objc_autoreleasePoolPush();
	const of_unichar_t *characters = [self characters];

	if (range.length > SIZE_MAX - range.location ||
	    range.location + range.length > _s->length)
		@throw [OFOutOfRangeException exception];

	memcpy(buffer, characters + range.location,
	    range.length * sizeof(of_unichar_t));

	objc_autoreleasePoolPop(pool);
}

- (of_range_t)rangeOfString: (OFString*)string
		    options: (int)options
		      range: (of_range_t)range
{
	const char *cString = [string UTF8String];
	size_t i, cStringLength = [string UTF8StringLength];
	size_t rangeLocation, rangeLength;

	if (range.length > SIZE_MAX - range.location ||
	    range.location + range.length > _s->length)
		@throw [OFOutOfRangeException exception];

	if (_s->isUTF8) {
		rangeLocation = of_string_utf8_get_position(
		    _s->cString, range.location, _s->cStringLength);
		rangeLength = of_string_utf8_get_position(
		    _s->cString + rangeLocation, range.length,
		    _s->cStringLength - rangeLocation);
	} else {
		rangeLocation = range.location;
		rangeLength = range.length;
	}

	if (cStringLength == 0)
		return of_range(0, 0);

	if (cStringLength > rangeLength)
		return of_range(OF_NOT_FOUND, 0);

	if (options & OF_STRING_SEARCH_BACKWARDS) {
		for (i = rangeLength - cStringLength;; i--) {
			if (memcmp(_s->cString + rangeLocation + i, cString,
			    cStringLength) == 0) {
				range.location += of_string_utf8_get_index(
				    _s->cString + rangeLocation, i);
				range.length = [string length];

				return range;
			}

			/* Did not match and we're at the last char */
			if (i == 0)
				return of_range(OF_NOT_FOUND, 0);
		}
	} else {
		for (i = 0; i <= rangeLength - cStringLength; i++) {
			if (memcmp(_s->cString + rangeLocation + i, cString,
			    cStringLength) == 0) {
				range.location += of_string_utf8_get_index(
				    _s->cString + rangeLocation, i);
				range.length = [string length];

				return range;
			}
		}
	}

	return of_range(OF_NOT_FOUND, 0);
}

- (bool)containsString: (OFString*)string
{
	const char *cString = [string UTF8String];
	size_t i, cStringLength = [string UTF8StringLength];

	if (cStringLength == 0)
		return true;

	if (cStringLength > _s->cStringLength)
		return false;

	for (i = 0; i <= _s->cStringLength - cStringLength; i++)
		if (memcmp(_s->cString + i, cString, cStringLength) == 0)
			return true;

	return false;
}

- (OFString*)substringWithRange: (of_range_t)range
{
	size_t start = range.location;
	size_t end = range.location + range.length;

	if (range.length > SIZE_MAX - range.location || end > _s->length)
		@throw [OFOutOfRangeException exception];

	if (_s->isUTF8) {
		start = of_string_utf8_get_position(_s->cString, start,
		    _s->cStringLength);
		end = of_string_utf8_get_position(_s->cString, end,
		    _s->cStringLength);
	}

	return [OFString stringWithUTF8String: _s->cString + start
				       length: end - start];
}

- (bool)hasPrefix: (OFString*)prefix
{
	size_t cStringLength = [prefix UTF8StringLength];

	if (cStringLength > _s->cStringLength)
		return false;

	return (memcmp(_s->cString, [prefix UTF8String], cStringLength) == 0);
}

- (bool)hasSuffix: (OFString*)suffix
{
	size_t cStringLength = [suffix UTF8StringLength];

	if (cStringLength > _s->cStringLength)
		return false;

	return (memcmp(_s->cString + (_s->cStringLength - cStringLength),
	    [suffix UTF8String], cStringLength) == 0);
}

- (OFArray*)componentsSeparatedByString: (OFString*)delimiter
				options: (int)options
{
	void *pool;
	OFMutableArray *array;
	const char *cString = [delimiter UTF8String];
	size_t cStringLength = [delimiter UTF8StringLength];
	bool skipEmpty = (options & OF_STRING_SKIP_EMPTY);
	size_t i, last;
	OFString *component;

	array = [OFMutableArray array];
	pool = objc_autoreleasePoolPush();

	if (cStringLength > _s->cStringLength) {
		[array addObject: [[self copy] autorelease]];
		objc_autoreleasePoolPop(pool);

		return array;
	}

	for (i = 0, last = 0; i <= _s->cStringLength - cStringLength; i++) {
		if (memcmp(_s->cString + i, cString, cStringLength) != 0)
			continue;

		component = [OFString stringWithUTF8String: _s->cString + last
						    length: i - last];
		if (!skipEmpty || [component length] > 0)
			[array addObject: component];

		i += cStringLength - 1;
		last = i + 1;
	}
	component = [OFString stringWithUTF8String: _s->cString + last];
	if (!skipEmpty || [component length] > 0)
		[array addObject: component];

	[array makeImmutable];

	objc_autoreleasePoolPop(pool);

	return array;
}

- (OFArray*)pathComponents
{
	OFMutableArray *ret;
	void *pool;
	size_t i, last = 0, pathCStringLength = _s->cStringLength;

	ret = [OFMutableArray array];

	if (pathCStringLength == 0)
		return ret;

	pool = objc_autoreleasePoolPush();

	if (OF_IS_PATH_DELIMITER(_s->cString[pathCStringLength - 1]))
		pathCStringLength--;

	for (i = 0; i < pathCStringLength; i++) {
		if (OF_IS_PATH_DELIMITER(_s->cString[i])) {
			[ret addObject:
			    [OFString stringWithUTF8String: _s->cString + last
						    length: i - last]];
			last = i + 1;
		}
	}

	[ret addObject: [OFString stringWithUTF8String: _s->cString + last
						length: i - last]];

	[ret makeImmutable];

	objc_autoreleasePoolPop(pool);

	return ret;
}

- (OFString*)lastPathComponent
{
	size_t pathCStringLength = _s->cStringLength;
	ssize_t i;

	if (pathCStringLength == 0)
		return @"";

	if (OF_IS_PATH_DELIMITER(_s->cString[pathCStringLength - 1]))
		pathCStringLength--;

	if (pathCStringLength == 0)
		return @"";

	if (pathCStringLength - 1 > SSIZE_MAX)
		@throw [OFOutOfRangeException exception];

	for (i = pathCStringLength - 1; i >= 0; i--) {
		if (OF_IS_PATH_DELIMITER(_s->cString[i])) {
			i++;
			break;
		}
	}

	/*
	 * Only one component, but the trailing delimiter might have been
	 * removed, so return a new string anyway.
	 */
	if (i < 0)
		i = 0;

	return [OFString stringWithUTF8String: _s->cString + i
				       length: pathCStringLength - i];
}

- (OFString*)stringByDeletingLastPathComponent
{
	size_t i, pathCStringLength = _s->cStringLength;

	if (pathCStringLength == 0)
		return @"";

	if (OF_IS_PATH_DELIMITER(_s->cString[pathCStringLength - 1]))
		pathCStringLength--;

	if (pathCStringLength == 0)
		return [OFString stringWithUTF8String: _s->cString
					       length: 1];

	for (i = pathCStringLength - 1; i >= 1; i--)
		if (OF_IS_PATH_DELIMITER(_s->cString[i]))
			return [OFString stringWithUTF8String: _s->cString
						       length: i];

	if (OF_IS_PATH_DELIMITER(_s->cString[0]))
		return [OFString stringWithUTF8String: _s->cString
					       length: 1];

	return @".";
}

- (const of_unichar_t*)characters
{
	OFObject *object = [[[OFObject alloc] init] autorelease];
	of_unichar_t *ret;
	size_t i, j;

	ret = [object allocMemoryWithSize: sizeof(of_unichar_t)
				    count: _s->length];

	i = j = 0;

	while (i < _s->cStringLength) {
		of_unichar_t c;
		size_t cLen;

		cLen = of_string_utf8_decode(_s->cString + i,
		    _s->cStringLength - i, &c);

		if (cLen == 0 || c > 0x10FFFF)
			@throw [OFInvalidEncodingException exception];

		ret[j++] = c;
		i += cLen;
	}

	return ret;
}

- (const of_char32_t*)UTF32StringWithByteOrder: (of_byte_order_t)byteOrder
{
	OFObject *object = [[[OFObject alloc] init] autorelease];
	of_char32_t *ret;
	size_t i, j;

	ret = [object allocMemoryWithSize: sizeof(of_unichar_t)
				    count: _s->length + 1];

	i = j = 0;

	while (i < _s->cStringLength) {
		of_unichar_t c;
		size_t cLen;

		cLen = of_string_utf8_decode(_s->cString + i,
		    _s->cStringLength - i, &c);

		if (cLen == 0 || c > 0x10FFFF)
			@throw [OFInvalidEncodingException exception];

		if (byteOrder != OF_BYTE_ORDER_NATIVE)
			ret[j++] = OF_BSWAP32(c);
		else
			ret[j++] = c;

		i += cLen;
	}
	ret[j] = 0;

	return ret;
}

#ifdef OF_HAVE_BLOCKS
- (void)enumerateLinesUsingBlock: (of_string_line_enumeration_block_t)block
{
	void *pool;
	const char *cString = _s->cString;
	const char *last = cString;
	bool stop = false, lastCarriageReturn = false;

	while (!stop && *cString != 0) {
		if (lastCarriageReturn && *cString == '\n') {
			lastCarriageReturn = false;

			cString++;
			last++;

			continue;
		}

		if (*cString == '\n' || *cString == '\r') {
			pool = objc_autoreleasePoolPush();

			block([OFString
			    stringWithUTF8String: last
					  length: cString - last], &stop);
			last = cString + 1;

			objc_autoreleasePoolPop(pool);
		}

		lastCarriageReturn = (*cString == '\r');
		cString++;
	}

	pool = objc_autoreleasePoolPush();

	if (!stop)
		block([OFString stringWithUTF8String: last
					      length: cString - last], &stop);

	objc_autoreleasePoolPop(pool);
}
#endif
@end
