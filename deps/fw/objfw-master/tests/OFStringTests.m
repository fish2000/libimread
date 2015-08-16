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
#include <string.h>
#include <math.h>

#import "OFString.h"
#import "OFMutableString_UTF8.h"
#import "OFArray.h"
#import "OFURL.h"
#import "OFAutoreleasePool.h"

#import "OFInvalidArgumentException.h"
#import "OFInvalidEncodingException.h"
#import "OFInvalidFormatException.h"
#import "OFOutOfRangeException.h"
#import "OFUnknownXMLEntityException.h"

#import "TestsAppDelegate.h"

static OFString *module = @"OFString";
static OFString* whitespace[] = {
	@" \r \t\n\t \tasd  \t \t\t\r\n",
	@" \t\t  \t\t  \t \t"
};
static of_unichar_t ucstr[] = {
	0xFEFF, 'f', 0xF6, 0xF6, 'b', 0xE4, 'r', 0x1F03A, 0
};
static of_unichar_t sucstr[] = {
	0xFFFE0000, 0x66000000, 0xF6000000, 0xF6000000, 0x62000000, 0xE4000000,
	0x72000000, 0x3AF00100, 0
};
static uint16_t utf16str[] = {
	0xFEFF, 'f', 0xF6, 0xF6, 'b', 0xE4, 'r', 0xD83C, 0xDC3A, 0
};
static uint16_t sutf16str[] = {
	0xFFFE, 0x6600, 0xF600, 0xF600, 0x6200, 0xE400, 0x7200, 0x3CD8, 0x3ADC,
	0
};

@interface EntityHandler: OFObject <OFStringXMLUnescapingDelegate>
@end

@implementation EntityHandler
-	   (OFString*)string: (OFString*)string
  containsUnknownEntityNamed: (OFString*)entity
{
	if ([entity isEqual: @"foo"])
		return @"bar";

	return nil;
}
@end

@implementation TestsAppDelegate (OFStringTests)
- (void)stringTests
{
	OFAutoreleasePool *pool = [[OFAutoreleasePool alloc] init];
	OFMutableString *s[3];
	OFString *is;
	OFArray *a;
	int i;
	const of_unichar_t *ua;
	const uint16_t *u16a;
	EntityHandler *h;
#ifdef OF_HAVE_BLOCKS
	__block bool ok;
#endif

	s[0] = [OFMutableString stringWithString: @"täs€"];
	s[1] = [OFMutableString string];
	s[2] = [[s[0] copy] autorelease];

	TEST(@"-[isEqual:]", [s[0] isEqual: s[2]] &&
	    ![s[0] isEqual: [[[OFObject alloc] init] autorelease]])

	TEST(@"-[compare:]", [s[0] compare: s[2]] == OF_ORDERED_SAME &&
	    [s[0] compare: @""] != OF_ORDERED_SAME &&
	    [@"" compare: @"a"] == OF_ORDERED_ASCENDING &&
	    [@"a" compare: @"b"] == OF_ORDERED_ASCENDING &&
	    [@"cd" compare: @"bc"] == OF_ORDERED_DESCENDING &&
	    [@"ä" compare: @"ö"] == OF_ORDERED_ASCENDING &&
	    [@"€" compare: @"ß"] == OF_ORDERED_DESCENDING &&
	    [@"aa" compare: @"z"] == OF_ORDERED_ASCENDING)

	TEST(@"-[caseInsensitiveCompare:]",
	    [@"a" caseInsensitiveCompare: @"A"] == OF_ORDERED_SAME &&
	    [@"Ä" caseInsensitiveCompare: @"ä"] == OF_ORDERED_SAME &&
	    [@"я" caseInsensitiveCompare: @"Я"] == OF_ORDERED_SAME &&
	    [@"€" caseInsensitiveCompare: @"ß"] == OF_ORDERED_DESCENDING &&
	    [@"ß" caseInsensitiveCompare: @"→"] == OF_ORDERED_ASCENDING &&
	    [@"AA" caseInsensitiveCompare: @"z"] == OF_ORDERED_ASCENDING &&
	    [[OFString stringWithUTF8String: "ABC"] caseInsensitiveCompare:
	    [OFString stringWithUTF8String: "AbD"]] == [@"abc" compare: @"abd"])

	TEST(@"-[hash] is the same if -[isEqual:] is true",
	    [s[0] hash] == [s[2] hash])

	TEST(@"-[description]", [[s[0] description] isEqual: s[0]])

	TEST(@"-[appendString:] and -[appendUTF8String:]",
	    R([s[1] appendUTF8String: "1𝄞"]) && R([s[1] appendString: @"3"]) &&
	    R([s[0] appendString: s[1]]) && [s[0] isEqual: @"täs€1𝄞3"])

	TEST(@"-[appendCharacters:length:]",
	    R([s[1] appendCharacters: ucstr + 6
			      length: 2]) && [s[1] isEqual: @"1𝄞3r🀺"])

	TEST(@"-[length]", [s[0] length] == 7)
	TEST(@"-[UTF8StringLength]", [s[0] UTF8StringLength] == 13)
	TEST(@"-[hash]", [s[0] hash] == 0x705583C0)

	TEST(@"-[characterAtIndex:]", [s[0] characterAtIndex: 0] == 't' &&
	    [s[0] characterAtIndex: 1] == 0xE4 &&
	    [s[0] characterAtIndex: 3] == 0x20AC &&
	    [s[0] characterAtIndex: 5] == 0x1D11E)

	EXPECT_EXCEPTION(@"Detect out of range in -[characterAtIndex:]",
	    OFOutOfRangeException, [s[0] characterAtIndex: 7])

	TEST(@"-[reverse]", R([s[0] reverse]) && [s[0] isEqual: @"3𝄞1€sät"])

	s[1] = [OFMutableString stringWithString: @"abc"];

	TEST(@"-[uppercase]", R([s[0] uppercase]) &&
	    [s[0] isEqual: @"3𝄞1€SÄT"] &&
	    R([s[1] uppercase]) && [s[1] isEqual: @"ABC"])

	TEST(@"-[lowercase]", R([s[0] lowercase]) &&
	    [s[0] isEqual: @"3𝄞1€sät"] &&
	    R([s[1] lowercase]) && [s[1] isEqual: @"abc"])

	TEST(@"-[uppercaseString]",
	    [[s[0] uppercaseString] isEqual: @"3𝄞1€SÄT"])

	TEST(@"-[lowercaseString]", R([s[0] uppercase]) &&
	    [[s[0] lowercaseString] isEqual: @"3𝄞1€sät"])

	TEST(@"-[capitalizedString]", [[@"ǆbla tǆst TǄST" capitalizedString]
	    isEqual: @"ǅbla Tǆst Tǆst"])

	TEST(@"+[stringWithUTF8String:length:]",
	    (s[0] = [OFMutableString stringWithUTF8String: "\xEF\xBB\xBF"
							   "foobar"
						   length: 6]) &&
	    [s[0] isEqual: @"foo"])

	TEST(@"+[stringWithUTF16String:]",
	    (is = [OFString stringWithUTF16String: utf16str]) &&
	    [is isEqual: @"fööbär🀺"] &&
	    (is = [OFString stringWithUTF16String: sutf16str]) &&
	    [is isEqual: @"fööbär🀺"])

	TEST(@"+[stringWithUTF32String:]",
	    (is = [OFString stringWithUTF32String: ucstr]) &&
	    [is isEqual: @"fööbär🀺"] &&
	    (is = [OFString stringWithUTF32String: sucstr]) &&
	    [is isEqual: @"fööbär🀺"])

#ifdef OF_HAVE_FILES
	TEST(@"+[stringWithContentsOfFile:encoding]", (is = [OFString
	    stringWithContentsOfFile: @"testfile.txt"
			    encoding: OF_STRING_ENCODING_ISO_8859_1]) &&
	    [is isEqual: @"testäöü"])

	TEST(@"+[stringWithContentsOfURL:encoding]", (is = [OFString
	    stringWithContentsOfURL: [OFURL URLWithString:
					 @"file://testfile.txt"]
			   encoding: OF_STRING_ENCODING_ISO_8859_1]) &&
	    [is isEqual: @"testäöü"])
#endif

	TEST(@"-[appendUTFString:length:]",
	    R([s[0] appendUTF8String: "foo\xEF\xBB\xBF" "barqux" + 3
			      length: 6]) && [s[0] isEqual: @"foobar"])

	EXPECT_EXCEPTION(@"Detection of invalid UTF-8 encoding #1",
	    OFInvalidEncodingException,
	    [OFString stringWithUTF8String: "\xE0\x80"])
	EXPECT_EXCEPTION(@"Detection of invalid UTF-8 encoding #2",
	    OFInvalidEncodingException,
	    [OFString stringWithUTF8String: "\xF0\x80\x80\xC0"])

	TEST(@"-[reverse] on UTF-8 strings",
	    (s[0] = [OFMutableString_UTF8 stringWithUTF8String: "äöü€𝄞"]) &&
	    R([s[0] reverse]) && [s[0] isEqual: @"𝄞€üöä"])

	TEST(@"Conversion of ISO 8859-1 to Unicode",
	    [[OFString stringWithCString: "\xE4\xF6\xFC"
				encoding: OF_STRING_ENCODING_ISO_8859_1]
	    isEqual: @"äöü"])

	TEST(@"Conversion of ISO 8859-15 to Unicode",
	    [[OFString stringWithCString: "\xA4\xA6\xA8\xB4\xB8\xBC\xBD\xBE"
				encoding: OF_STRING_ENCODING_ISO_8859_15]
	    isEqual: @"€ŠšŽžŒœŸ"])

	TEST(@"Conversion of Windows 1252 to Unicode",
	    [[OFString stringWithCString: "\x80\x82\x83\x84\x85\x86\x87\x88"
					  "\x89\x8A\x8B\x8C\x8E\x91\x92\x93"
					  "\x94\x95\x96\x97\x98\x99\x9A\x9B"
					  "\x9C\x9E\x9F"
				encoding: OF_STRING_ENCODING_WINDOWS_1252]
	    isEqual: @"€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ"])

	TEST(@"Conversion of Codepage 437 to Unicode",
	    [[OFString stringWithCString: "\xB0\xB1\xB2\xDB"
				encoding: OF_STRING_ENCODING_CODEPAGE_437]
	    isEqual: @"░▒▓█"])

	TEST(@"Conversion of Unicode to ASCII #1",
	    !strcmp([@"This is a test" cStringWithEncoding:
	    OF_STRING_ENCODING_ASCII], "This is a test"))

	EXPECT_EXCEPTION(@"Conversion of Unicode to ASCII #2",
	    OFInvalidEncodingException,
	    [@"This is a tést" cStringWithEncoding: OF_STRING_ENCODING_ASCII])

	TEST(@"Conversion of Unicode to ISO-8859-1 #1",
	    !strcmp([@"This is ä test" cStringWithEncoding:
	    OF_STRING_ENCODING_ISO_8859_1], "This is \xE4 test"))

	EXPECT_EXCEPTION(@"Conversion of Unicode to ISO-8859-1 #2",
	    OFInvalidEncodingException, [@"This is ä t€st" cStringWithEncoding:
	    OF_STRING_ENCODING_ISO_8859_1])

	TEST(@"Conversion of Unicode to ISO-8859-15 #1",
	    !strcmp([@"This is ä t€st" cStringWithEncoding:
	    OF_STRING_ENCODING_ISO_8859_15], "This is \xE4 t\xA4st"))

	EXPECT_EXCEPTION(@"Conversion of Unicode to ISO-8859-15 #2",
	    OFInvalidEncodingException, [@"This is ä t€st…" cStringWithEncoding:
	    OF_STRING_ENCODING_ISO_8859_15])

	TEST(@"Conversion of Unicode to Windows-1252 #1",
	    !strcmp([@"This is ä t€st…" cStringWithEncoding:
	    OF_STRING_ENCODING_WINDOWS_1252], "This is \xE4 t\x80st\x85"))

	EXPECT_EXCEPTION(@"Conversion of Unicode to Windows-1252 #2",
	    OFInvalidEncodingException, [@"This is ä t€st…‼"
	    cStringWithEncoding: OF_STRING_ENCODING_WINDOWS_1252])

	TEST(@"Conversion of Unicode to Codepage 437 #1",
	    !strcmp([@"Tést strîng ░▒▓" cStringWithEncoding:
	    OF_STRING_ENCODING_CODEPAGE_437], "T\x82st str\x8Cng \xB0\xB1\xB2"))

	EXPECT_EXCEPTION(@"Conversion of Unicode to Codepage 437 #2",
	    OFInvalidEncodingException, [@"T€st strîng ░▒▓"
	    cStringWithEncoding: OF_STRING_ENCODING_CODEPAGE_437])

	TEST(@"Lossy conversion of Unicode to ASCII",
	    !strcmp([@"This is a tést" lossyCStringWithEncoding:
	    OF_STRING_ENCODING_ASCII], "This is a t?st"))

	TEST(@"Lossy conversion of Unicode to ISO-8859-1",
	    !strcmp([@"This is ä t€st" lossyCStringWithEncoding:
	    OF_STRING_ENCODING_ISO_8859_1], "This is \xE4 t?st"))

	TEST(@"Lossy conversion of Unicode to ISO-8859-15",
	    !strcmp([@"This is ä t€st…" lossyCStringWithEncoding:
	    OF_STRING_ENCODING_ISO_8859_15], "This is \xE4 t\xA4st?"))

	TEST(@"Lossy conversion of Unicode to Windows-1252",
	    !strcmp([@"This is ä t€st…‼" lossyCStringWithEncoding:
	    OF_STRING_ENCODING_WINDOWS_1252], "This is \xE4 t\x80st\x85?"))

	TEST(@"Lossy conversion of Unicode to Codepage 437",
	    !strcmp([@"T€st strîng ░▒▓" lossyCStringWithEncoding:
	    OF_STRING_ENCODING_CODEPAGE_437], "T?st str\x8Cng \xB0\xB1\xB2"))

	TEST(@"+[stringWithFormat:]",
	    [(s[0] = [OFMutableString stringWithFormat: @"%@:%d", @"test", 123])
	    isEqual: @"test:123"])

	TEST(@"-[appendFormat:]",
	    R(([s[0] appendFormat: @"%02X", 15])) &&
	    [s[0] isEqual: @"test:1230F"])

	TEST(@"-[rangeOfString:]",
	    [@"𝄞öö" rangeOfString: @"öö"].location == 1 &&
	    [@"𝄞öö" rangeOfString: @"ö"].location == 1 &&
	    [@"𝄞öö" rangeOfString: @"𝄞"].location == 0 &&
	    [@"𝄞öö" rangeOfString: @"x"].location == OF_NOT_FOUND &&
	    [@"𝄞öö" rangeOfString: @"öö"
			  options: OF_STRING_SEARCH_BACKWARDS].location == 1 &&
	    [@"𝄞öö" rangeOfString: @"ö"
			  options: OF_STRING_SEARCH_BACKWARDS].location == 2 &&
	    [@"𝄞öö" rangeOfString: @"𝄞"
			  options: OF_STRING_SEARCH_BACKWARDS].location == 0 &&
	    [@"𝄞öö" rangeOfString: @"x"
			  options: OF_STRING_SEARCH_BACKWARDS].location ==
	     OF_NOT_FOUND)

	TEST(@"-[substringWithRange:]",
	    [[@"𝄞öö" substringWithRange: of_range(1, 1)] isEqual: @"ö"] &&
	    [[@"𝄞öö" substringWithRange: of_range(3, 0)] isEqual: @""])

	EXPECT_EXCEPTION(@"Detect out of range in -[substringWithRange:] #1",
	    OFOutOfRangeException, [@"𝄞öö" substringWithRange: of_range(2, 2)])
	EXPECT_EXCEPTION(@"Detect out of range in -[substringWithRange:] #2",
	    OFOutOfRangeException, [@"𝄞öö" substringWithRange: of_range(4, 0)])

	TEST(@"-[stringByAppendingString:]",
	    [[@"foo" stringByAppendingString: @"bar"] isEqual: @"foobar"])

	TEST(@"-[stringByPrependingString:]",
	    [[@"foo" stringByPrependingString: @"bar"] isEqual: @"barfoo"])

	TEST(@"-[hasPrefix:]", [@"foobar" hasPrefix: @"foo"] &&
	    ![@"foobar" hasPrefix: @"foobar0"])

	TEST(@"-[hasSuffix:]", [@"foobar" hasSuffix: @"bar"] &&
	    ![@"foobar" hasSuffix: @"foobar0"])

	i = 0;
	TEST(@"-[componentsSeparatedByString:]",
	    (a = [@"fooXXbarXXXXbazXXXX" componentsSeparatedByString: @"XX"]) &&
	    [a count] == 6 &&
	    [[a objectAtIndex: i++] isEqual: @"foo"] &&
	    [[a objectAtIndex: i++] isEqual: @"bar"] &&
	    [[a objectAtIndex: i++] isEqual: @""] &&
	    [[a objectAtIndex: i++] isEqual: @"baz"] &&
	    [[a objectAtIndex: i++] isEqual: @""] &&
	    [[a objectAtIndex: i++] isEqual: @""])

	i = 0;
	TEST(@"-[componentsSeparatedByString:options:]",
	    (a = [@"fooXXbarXXXXbazXXXX"
	    componentsSeparatedByString: @"XX"
				options: OF_STRING_SKIP_EMPTY]) &&
	    [a count] == 3 &&
	    [[a objectAtIndex: i++] isEqual: @"foo"] &&
	    [[a objectAtIndex: i++] isEqual: @"bar"] &&
	    [[a objectAtIndex: i++] isEqual: @"baz"])

#if !defined(_WIN32) && !defined(__DJGPP__)
# define EXPECTED @"foo/bar/baz"
#else
# define EXPECTED @"foo\\bar\\baz"
#endif
	TEST(@"+[pathWithComponents:]",
	    (is = [OFString pathWithComponents: [OFArray arrayWithObjects:
	    @"foo", @"bar", @"baz", nil]]) &&
	    [is isEqual: EXPECTED] &&
	    (is = [OFString pathWithComponents:
	    [OFArray arrayWithObjects: @"foo", nil]]) &&
	    [is isEqual: @"foo"])
#undef EXPECTED

	TEST(@"-[pathComponents]",
	    /* /tmp */
	    (a = [@"/tmp" pathComponents]) && [a count] == 2 &&
	    [[a objectAtIndex: 0] isEqual: @""] &&
	    [[a objectAtIndex: 1] isEqual: @"tmp"] &&
	    /* /tmp/ */
	    (a = [@"/tmp/" pathComponents]) && [a count] == 2 &&
	    [[a objectAtIndex: 0] isEqual: @""] &&
	    [[a objectAtIndex: 1] isEqual: @"tmp"] &&
	    /* / */
	    (a = [@"/" pathComponents]) && [a count] == 1 &&
	    [[a objectAtIndex: 0] isEqual: @""] &&
	    /* foo/bar */
	    (a = [@"foo/bar" pathComponents]) && [a count] == 2 &&
	    [[a objectAtIndex: 0] isEqual: @"foo"] &&
	    [[a objectAtIndex: 1] isEqual: @"bar"] &&
	    /* foo/bar/baz/ */
	    (a = [@"foo/bar/baz" pathComponents]) && [a count] == 3 &&
	    [[a objectAtIndex: 0] isEqual: @"foo"] &&
	    [[a objectAtIndex: 1] isEqual: @"bar"] &&
	    [[a objectAtIndex: 2] isEqual: @"baz"] &&
	    /* foo// */
	    (a = [@"foo//" pathComponents]) && [a count] == 2 &&
	    [[a objectAtIndex: 0] isEqual: @"foo"] &&
	    [[a objectAtIndex: 1] isEqual: @""] &&
	    [[@"" pathComponents] count] == 0)

	TEST(@"-[lastPathComponent]",
	    [[@"/tmp" lastPathComponent] isEqual: @"tmp"] &&
	    [[@"/tmp/" lastPathComponent] isEqual: @"tmp"] &&
	    [[@"/" lastPathComponent] isEqual: @""] &&
	    [[@"foo" lastPathComponent] isEqual: @"foo"] &&
	    [[@"foo/bar" lastPathComponent] isEqual: @"bar"] &&
	    [[@"foo/bar/baz/" lastPathComponent] isEqual: @"baz"])

	TEST(@"-[pathExtension]",
	    [[@"foo.bar" pathExtension] isEqual: @"bar"] &&
	    [[@"foo/.bar" pathExtension] isEqual: @""] &&
	    [[@"foo/.bar.baz" pathExtension] isEqual: @"baz"] &&
	    [[@"foo/bar.baz/" pathExtension] isEqual: @"baz"])

	TEST(@"-[stringByDeletingLastPathComponent]",
	    [[@"/tmp" stringByDeletingLastPathComponent] isEqual: @"/"] &&
	    [[@"/tmp/" stringByDeletingLastPathComponent] isEqual: @"/"] &&
	    [[@"/tmp/foo/" stringByDeletingLastPathComponent]
	    isEqual: @"/tmp"] &&
	    [[@"foo/bar" stringByDeletingLastPathComponent] isEqual: @"foo"] &&
	    [[@"/" stringByDeletingLastPathComponent] isEqual: @"/"] &&
	    [[@"foo" stringByDeletingLastPathComponent] isEqual: @"."])

#if !defined(_WIN32) && !defined(__DJGPP__)
# define EXPECTED @"/foo./bar"
#else
# define EXPECTED @"\\foo.\\bar"
#endif
	TEST(@"-[stringByDeletingPathExtension]",
	    [[@"foo.bar" stringByDeletingPathExtension] isEqual: @"foo"] &&
	    [[@"foo..bar" stringByDeletingPathExtension] isEqual: @"foo."] &&
	    [[@"/foo./bar" stringByDeletingPathExtension]
	    isEqual: @"/foo./bar"] &&
	    [[@"/foo./bar.baz" stringByDeletingPathExtension]
	    isEqual: EXPECTED] &&
	    [[@"foo.bar/" stringByDeletingPathExtension] isEqual: @"foo"] &&
	    [[@".foo" stringByDeletingPathExtension] isEqual: @".foo"] &&
	    [[@".foo.bar" stringByDeletingPathExtension] isEqual: @".foo"])
#undef EXPECTED

	TEST(@"-[decimalValue]",
	    [@"1234" decimalValue] == 1234 &&
	    [@"\r\n+123  " decimalValue] == 123 &&
	    [@"-500\t" decimalValue] == -500 &&
	    [@"\t\t\r\n" decimalValue] == 0)

	TEST(@"-[hexadecimalValue]",
	    [@"123f" hexadecimalValue] == 0x123f &&
	    [@"\t\n0xABcd\r" hexadecimalValue] == 0xABCD &&
	    [@"  xbCDE" hexadecimalValue] == 0xBCDE &&
	    [@"$CdEf" hexadecimalValue] == 0xCDEF &&
	    [@"\rFeh " hexadecimalValue] == 0xFE &&
	    [@"\r\t" hexadecimalValue] == 0)

	/*
	 * These test numbers can be generated without rounding if we have IEEE
	 * floating point numbers, thus we can use == on them.
	 */
	TEST(@"-[floatValue]",
	    [@"\t-0.25 " floatValue] == -0.25 &&
	    [@"\r-INFINITY\n" floatValue] == -INFINITY &&
	    isnan([@"   NAN\t\t" floatValue]))

#if !defined(__ANDROID__) && !defined(__sun__) && !defined(__DJGPP__)
# define INPUT @"\t-0x1.FFFFFFFFFFFFFP-1020 "
# define EXPECTED -0x1.FFFFFFFFFFFFFP-1020
#else
/* Android, Solaris and DJGPP do not accept 0x for strtod() */
# if !defined(__sun__) || !defined(__i386__)
#  define INPUT @"\t-0.123456789 "
#  define EXPECTED -0.123456789
# else
/* Solaris' strtod() has weird rounding on x86, but not on x86_64 */
#  define INPUT @"\t-0.125 "
#  define EXPECTED -0.125
# endif
#endif
	TEST(@"-[doubleValue]",
	    [INPUT doubleValue] == EXPECTED &&
	    [@"\r-INFINITY\n" doubleValue] == -INFINITY &&
	    isnan([@"   NAN\t\t" doubleValue]))
#undef INPUT
#undef EXPECTED

	EXPECT_EXCEPTION(@"Detect invalid characters in -[decimalValue] #1",
	    OFInvalidFormatException, [@"abc" decimalValue])
	EXPECT_EXCEPTION(@"Detect invalid characters in -[decimalValue] #2",
	    OFInvalidFormatException, [@"0a" decimalValue])
	EXPECT_EXCEPTION(@"Detect invalid characters in -[decimalValue] #3",
	    OFInvalidFormatException, [@"0 1" decimalValue])

	EXPECT_EXCEPTION(@"Detect invalid chars in -[hexadecimalValue] #1",
	    OFInvalidFormatException, [@"0xABCDEFG" hexadecimalValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[hexadecimalValue] #2",
	    OFInvalidFormatException, [@"0x" hexadecimalValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[hexadecimalValue] #3",
	    OFInvalidFormatException, [@"$" hexadecimalValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[hexadecimalValue] #4",
	    OFInvalidFormatException, [@"$ " hexadecimalValue])

	EXPECT_EXCEPTION(@"Detect invalid chars in -[floatValue] #1",
	    OFInvalidFormatException, [@"0,0" floatValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[floatValue] #2",
	    OFInvalidFormatException, [@"0.0a" floatValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[floatValue] #3",
	    OFInvalidFormatException, [@"0 0" floatValue])

	EXPECT_EXCEPTION(@"Detect invalid chars in -[doubleValue] #1",
	    OFInvalidFormatException, [@"0,0" floatValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[doubleValue] #2",
	    OFInvalidFormatException, [@"0.0a" floatValue])
	EXPECT_EXCEPTION(@"Detect invalid chars in -[doubleValue] #3",
	    OFInvalidFormatException, [@"0 0" floatValue])

	EXPECT_EXCEPTION(@"Detect out of range in -[decimalValue]",
	    OFOutOfRangeException,
	    [@"12345678901234567890123456789012345678901234567890"
	     @"12345678901234567890123456789012345678901234567890"
	    decimalValue])

	EXPECT_EXCEPTION(@"Detect out of range in -[hexadecimalValue]",
	    OFOutOfRangeException,
	    [@"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF"
	     @"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF"
	    hexadecimalValue])

	TEST(@"-[characters]", (ua = [@"fööbär🀺" characters]) &&
	    !memcmp(ua, ucstr + 1, sizeof(ucstr) - 8))

#ifdef OF_BIG_ENDIAN
# define SWAPPED_BYTE_ORDER OF_BYTE_ORDER_LITTLE_ENDIAN
#else
# define SWAPPED_BYTE_ORDER OF_BYTE_ORDER_BIG_ENDIAN
#endif
	TEST(@"-[UTF16String]", (u16a = [@"fööbär🀺" UTF16String]) &&
	    !memcmp(u16a, utf16str + 1, of_string_utf16_length(utf16str) * 2) &&
	    (u16a = [@"fööbär🀺"
	    UTF16StringWithByteOrder: SWAPPED_BYTE_ORDER]) &&
	    !memcmp(u16a, sutf16str + 1, of_string_utf16_length(sutf16str) * 2))

	TEST(@"-[UTF16StringLength]", [@"fööbär🀺" UTF16StringLength] == 8)

	TEST(@"-[UTF32String]", (ua = [@"fööbär🀺" UTF32String]) &&
	    !memcmp(ua, ucstr + 1, of_string_utf32_length(ucstr) * 4) &&
	    (ua = [@"fööbär🀺" UTF32StringWithByteOrder: SWAPPED_BYTE_ORDER]) &&
	    !memcmp(ua, sucstr + 1, of_string_utf32_length(sucstr) * 4))
#undef SWAPPED_BYTE_ORDER

	TEST(@"-[MD5Hash]", [[@"asdfoobar" MD5Hash]
	    isEqual: @"184dce2ec49b5422c7cfd8728864db4c"])

	TEST(@"-[RIPEMD160Hash]", [[@"asdfoobar" RIPEMD160Hash]
	    isEqual: @"021d773b0fac06eb6755ca6aa58a580c980f7f13"])

	TEST(@"-[SHA1Hash]", [[@"asdfoobar" SHA1Hash]
	    isEqual: @"f5f81ac0a8b5cbfdc4585ec1ad32e7b3a12b9b49"])

	TEST(@"-[SHA224Hash]", [[@"asdfoobar" SHA224Hash]
	    isEqual:
	    @"5a06822dcbd5a874f67d062b80b9d8a9cb9b5b303960b9da9290c192"])

	TEST(@"-[SHA256Hash]", [[@"asdfoobar" SHA256Hash] isEqual:
	    @"28e65b1dcd7f6ce2ea6277b15f87b913"
	    @"628b5500bf7913a2bbf4cedcfa1215f6"])

	TEST(@"-[SHA384Hash]", [[@"asdfoobar" SHA384Hash] isEqual:
	    @"73286da882ffddca2f45e005cfa6b44f3fc65bfb26db1d08"
	    @"7ded2f9c279e5addf8be854044bca0cece073fce28eec7d9"])

	TEST(@"-[SHA512Hash]", [[@"asdfoobar" SHA512Hash] isEqual:
	    @"0464c427da158b02161bb44a3090bbfc594611ef6a53603640454b56412a9247c"
	    @"3579a329e53a5dc74676b106755e3394f9454a2d42273242615d32f80437d61"])

	TEST(@"-[stringByURLEncoding]",
	    [[@"foo\"ba'_~$" stringByURLEncoding] isEqual: @"foo%22ba%27_%7E$"])

	TEST(@"-[stringByURLDecoding]",
	    [[@"foo%20bar%22+%24" stringByURLDecoding] isEqual: @"foo bar\" $"])

	TEST(@"-[insertString:atIndex:]",
	    (s[0] = [OFMutableString stringWithString: @"𝄞öööbä€"]) &&
	    R([s[0] insertString: @"äöü"
			 atIndex: 3]) &&
	    [s[0] isEqual: @"𝄞ööäöüöbä€"])

	EXPECT_EXCEPTION(@"Detect invalid format in -[stringByURLDecoding] "
	    @"#1", OFInvalidFormatException, [@"foo%xbar" stringByURLDecoding])
	EXPECT_EXCEPTION(@"Detect invalid encoding in -[stringByURLDecoding] "
	    @"#2", OFInvalidEncodingException,
	    [@"foo%FFbar" stringByURLDecoding])

	TEST(@"-[setCharacter:atIndex:]",
	    (s[0] = [OFMutableString stringWithString: @"abäde"]) &&
	    R([s[0] setCharacter: 0xF6
			 atIndex: 2]) &&
	    [s[0] isEqual: @"aböde"] &&
	    R([s[0] setCharacter: 'c'
			 atIndex: 2]) &&
	    [s[0] isEqual: @"abcde"] &&
	    R([s[0] setCharacter: 0x20AC
			 atIndex: 3]) &&
	    [s[0] isEqual: @"abc€e"] &&
	    R([s[0] setCharacter: 'x'
			 atIndex: 1]) &&
	    [s[0] isEqual: @"axc€e"])

	TEST(@"-[deleteCharactersInRange:]",
	    (s[0] = [OFMutableString stringWithString: @"𝄞öööbä€"]) &&
	    R([s[0] deleteCharactersInRange: of_range(1, 3)]) &&
	    [s[0] isEqual: @"𝄞bä€"] &&
	    R([s[0] deleteCharactersInRange: of_range(0, 4)]) &&
	    [s[0] isEqual: @""])

	TEST(@"-[replaceCharactersInRange:withString:]",
	    (s[0] = [OFMutableString stringWithString: @"𝄞öööbä€"]) &&
	    R([s[0] replaceCharactersInRange: of_range(1, 3)
				  withString: @"äöüß"]) &&
	    [s[0] isEqual: @"𝄞äöüßbä€"] &&
	    R([s[0] replaceCharactersInRange: of_range(4, 2)
				  withString: @"b"]) &&
	    [s[0] isEqual: @"𝄞äöübä€"] &&
	    R([s[0] replaceCharactersInRange: of_range(0, 7)
				  withString: @""]) &&
	    [s[0] isEqual: @""])

	EXPECT_EXCEPTION(@"Detect OoR in -[deleteCharactersInRange:] #1",
	    OFOutOfRangeException,
	    {
		s[0] = [OFMutableString stringWithString: @"𝄞öö"];
		[s[0] deleteCharactersInRange: of_range(2, 2)];
	    })

	EXPECT_EXCEPTION(@"Detect OoR in -[deleteCharactersInRange:] #2",
	    OFOutOfRangeException,
	    [s[0] deleteCharactersInRange: of_range(4, 0)])

	EXPECT_EXCEPTION(@"Detect OoR in "
	    @"-[replaceCharactersInRange:withString:] #1",
	    OFOutOfRangeException,
	    [s[0] replaceCharactersInRange: of_range(2, 2)
				withString: @""])

	EXPECT_EXCEPTION(@"Detect OoR in "
	    @"-[replaceCharactersInRange:withString:] #2",
	    OFOutOfRangeException,
	    [s[0] replaceCharactersInRange: of_range(4, 0)
				withString: @""])

	TEST(@"-[replaceOccurrencesOfString:withString:]",
	    (s[0] = [OFMutableString stringWithString:
	    @"asd fo asd fofo asd"]) &&
	    R([s[0] replaceOccurrencesOfString: @"fo"
				    withString: @"foo"]) &&
	    [s[0] isEqual: @"asd foo asd foofoo asd"] &&
	    (s[0] = [OFMutableString stringWithString: @"XX"]) &&
	    R([s[0] replaceOccurrencesOfString: @"X"
				    withString: @"XX"]) &&
	    [s[0] isEqual: @"XXXX"])

	TEST(@"-[replaceOccurrencesOfString:withString:options:range:]",
	    (s[0] = [OFMutableString stringWithString:
	    @"foofoobarfoobarfoo"]) &&
	    R([s[0] replaceOccurrencesOfString: @"oo"
				    withString: @"óò"
				       options: 0
					 range: of_range(2, 15)]) &&
	    [s[0] isEqual: @"foofóòbarfóòbarfoo"])

	TEST(@"-[deleteLeadingWhitespaces]",
	    (s[0] = [OFMutableString stringWithString: whitespace[0]]) &&
	    R([s[0] deleteLeadingWhitespaces]) &&
	    [s[0] isEqual: @"asd  \t \t\t\r\n"] &&
	    (s[0] = [OFMutableString stringWithString: whitespace[1]]) &&
	    R([s[0] deleteLeadingWhitespaces]) && [s[0] isEqual: @""])

	TEST(@"-[deleteTrailingWhitespaces]",
	    (s[0] = [OFMutableString stringWithString: whitespace[0]]) &&
	    R([s[0] deleteTrailingWhitespaces]) &&
	    [s[0] isEqual: @" \r \t\n\t \tasd"] &&
	    (s[0] = [OFMutableString stringWithString: whitespace[1]]) &&
	    R([s[0] deleteTrailingWhitespaces]) && [s[0] isEqual: @""])

	TEST(@"-[deleteEnclosingWhitespaces]",
	    (s[0] = [OFMutableString stringWithString: whitespace[0]]) &&
	    R([s[0] deleteEnclosingWhitespaces]) && [s[0] isEqual: @"asd"] &&
	    (s[0] = [OFMutableString stringWithString: whitespace[1]]) &&
	    R([s[0] deleteEnclosingWhitespaces]) && [s[0] isEqual: @""])

	TEST(@"-[stringByXMLEscaping]",
	    (s[0] = (id)[@"<hello> &world'\"!&" stringByXMLEscaping]) &&
	    [s[0] isEqual: @"&lt;hello&gt; &amp;world&apos;&quot;!&amp;"])

	TEST(@"-[stringByXMLUnescaping]",
	    [[s[0] stringByXMLUnescaping] isEqual: @"<hello> &world'\"!&"] &&
	    [[@"&#x79;" stringByXMLUnescaping] isEqual: @"y"] &&
	    [[@"&#xe4;" stringByXMLUnescaping] isEqual: @"ä"] &&
	    [[@"&#8364;" stringByXMLUnescaping] isEqual: @"€"] &&
	    [[@"&#x1D11E;" stringByXMLUnescaping] isEqual: @"𝄞"])

	EXPECT_EXCEPTION(@"Detect unknown entities in -[stringByXMLUnescaping]",
	    OFUnknownXMLEntityException, [@"&foo;" stringByXMLUnescaping])
	EXPECT_EXCEPTION(@"Detect invalid entities in -[stringByXMLUnescaping] "
	    @"#1", OFInvalidFormatException, [@"x&amp" stringByXMLUnescaping])
	EXPECT_EXCEPTION(@"Detect invalid entities in -[stringByXMLUnescaping] "
	    @"#2", OFInvalidFormatException, [@"&#;" stringByXMLUnescaping])
	EXPECT_EXCEPTION(@"Detect invalid entities in -[stringByXMLUnescaping] "
	    @"#3", OFInvalidFormatException, [@"&#x;" stringByXMLUnescaping])
	EXPECT_EXCEPTION(@"Detect invalid entities in -[stringByXMLUnescaping] "
	    @"#4", OFInvalidFormatException, [@"&#g;" stringByXMLUnescaping])
	EXPECT_EXCEPTION(@"Detect invalid entities in -[stringByXMLUnescaping] "
	    @"#5", OFInvalidFormatException, [@"&#xg;" stringByXMLUnescaping])

	TEST(@"-[stringByXMLUnescapingWithDelegate:]",
	    (h = [[[EntityHandler alloc] init] autorelease]) &&
	    [[@"x&foo;y" stringByXMLUnescapingWithDelegate: h]
	    isEqual: @"xbary"])

#ifdef OF_HAVE_BLOCKS
	TEST(@"-[stringByXMLUnescapingWithBlock:]",
	    [[@"x&foo;y" stringByXMLUnescapingWithBlock:
	        ^ OFString* (OFString *str, OFString *entity) {
		    if ([entity isEqual: @"foo"])
			    return @"bar";

		    return nil;
	    }] isEqual: @"xbary"])

	ok = true;
	[@"foo\nbar\nbaz" enumerateLinesUsingBlock:
	    ^ (OFString *line, bool *stop) {
		static int i = 0;

		switch (i) {
		case 0:
			if (![line isEqual: @"foo"])
				ok = false;
			break;
		case 1:
			if (![line isEqual: @"bar"])
				ok = false;
			break;
		case 2:
			if (![line isEqual: @"baz"])
				ok = false;
			break;
		default:
			ok = false;
		}

		i++;
	}];
	TEST(@"-[enumerateLinesUsingBlock:]", ok)
#endif

	[pool drain];
}
@end
