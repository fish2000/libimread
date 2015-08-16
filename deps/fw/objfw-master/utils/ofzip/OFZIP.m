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

#include <string.h>

#import "OFApplication.h"
#import "OFArray.h"
#import "OFDate.h"
#import "OFFile.h"
#import "OFOptionsParser.h"
#import "OFSet.h"
#import "OFStdIOStream.h"
#import "OFZIPArchive.h"
#import "OFZIPArchiveEntry.h"

#import "OFCreateDirectoryFailedException.h"
#import "OFInvalidFormatException.h"
#import "OFOpenItemFailedException.h"
#import "OFReadFailedException.h"
#import "OFWriteFailedException.h"

#define BUFFER_SIZE 4096

#ifndef S_IRWXG
# define S_IRWXG 0
#endif
#ifndef S_IRWXO
# define S_IRWXO 0
#endif

@interface OFZIP: OFObject
{
	int_fast8_t _override, _outputLevel;
	int _exitStatus;
}

- (OFZIPArchive*)openArchiveWithPath: (OFString*)path;
- (void)listFilesInArchive: (OFZIPArchive*)archive;
- (void)extractFiles: (OFArray OF_GENERIC(OFString*)*)files
	 fromArchive: (OFZIPArchive*)archive;
@end

OF_APPLICATION_DELEGATE(OFZIP)

static void
help(OFStream *stream, bool full, int status)
{
	[stream writeFormat:
	    @"Usage: %@ -[fhlnqvx] archive.zip [file1 file2 ...]\n",
	    [OFApplication programName]];

	if (full)
		[stream writeString:
		    @"\nOptions:\n"
		    @"    -f  Force / override files\n"
		    @"    -h  Show this help\n"
		    @"    -l  List all files in the archive\n"
		    @"    -n  Never override files\n"
		    @"    -q  Quiet mode (no output, except errors)\n"
		    @"    -v  Verbose output for file list\n"
		    @"    -x  Extract files\n"];

	[OFApplication terminateWithStatus: status];
}

static void
mutuallyExclusiveError(of_unichar_t option1, of_unichar_t option2)
{
	[of_stderr writeFormat: @"Error: -%C and -%C are mutually exclusive!\n",
				option1, option2];
	[OFApplication terminateWithStatus: 1];
}

static void
setPermissions(OFString *path, OFZIPArchiveEntry *entry)
{
#ifdef OF_HAVE_CHMOD
	if (([entry versionMadeBy] >> 8) ==
	    OF_ZIP_ARCHIVE_ENTRY_ATTR_COMPAT_UNIX) {
		uint32_t mode = [entry versionSpecificAttributes] >> 16;

		/* Only allow modes that are safe */
		mode &= (S_IRWXU | S_IRWXG | S_IRWXO);

		[OFFile changePermissionsOfItemAtPath: path
					  permissions: mode];
	}
#endif
}

@implementation OFZIP
- (void)applicationDidFinishLaunching
{
	OFOptionsParser *optionsParser =
	    [OFOptionsParser parserWithOptions: @"fhlnqvx"];
	of_unichar_t option, mode = '\0';
	OFArray OF_GENERIC(OFString*) *remainingArguments, *files;
	OFZIPArchive *archive;

	while ((option = [optionsParser nextOption]) != '\0') {
		switch (option) {
		case 'f':
			if (_override < 0)
				mutuallyExclusiveError('f', 'n');

			_override = 1;
			break;
		case 'n':
			if (_override > 0)
				mutuallyExclusiveError('f', 'n');

			_override = -1;
			break;
		case 'v':
			if (_outputLevel < 0)
				mutuallyExclusiveError('v', 'q');

			_outputLevel++;
			break;
		case 'q':
			if (_outputLevel > 0)
				mutuallyExclusiveError('v', 'q');

			_outputLevel--;
			break;
		case 'l':
		case 'x':
			if (mode != '\0')
				mutuallyExclusiveError('x', 'l');

			mode = option;
			break;
		case 'h':
			help(of_stdout, true, 0);
			break;
		default:
			[of_stderr writeFormat: @"%@: Unknown option: -%C\n",
						[OFApplication programName],
						[optionsParser lastOption]];
			[OFApplication terminateWithStatus: 1];
		}
	}

	remainingArguments = [optionsParser remainingArguments];

	switch (mode) {
	case 'l':
		if ([remainingArguments count] != 1)
			help(of_stderr, false, 1);

		archive = [self openArchiveWithPath:
		    [remainingArguments firstObject]];

		[self listFilesInArchive: archive];
		break;
	case 'x':
		if ([remainingArguments count] < 1)
			help(of_stderr, false, 1);

		files = [remainingArguments objectsInRange:
		    of_range(1, [remainingArguments count] - 1)];
		archive = [self openArchiveWithPath:
		    [remainingArguments firstObject]];

		@try {
			[self extractFiles: files
			       fromArchive: archive];
		} @catch (OFCreateDirectoryFailedException *e) {
			[of_stderr writeFormat:
			    @"\rFailed to create directory %@: %s\n",
			    [e path], strerror([e errNo])];
			_exitStatus = 1;
		} @catch (OFOpenItemFailedException *e) {
			[of_stderr writeFormat:
			    @"\rFailed to open file %@: %s\n",
			    [e path], strerror([e errNo])];
			_exitStatus = 1;
		}

		break;
	default:
		help(of_stderr, true, 1);
		break;
	}

	[OFApplication terminateWithStatus: _exitStatus];
}

- (OFZIPArchive*)openArchiveWithPath: (OFString*)path
{
	OFZIPArchive *archive = nil;

	@try {
		archive = [OFZIPArchive archiveWithPath: path];
	} @catch (OFOpenItemFailedException *e) {
		[of_stderr writeFormat: @"Failed to open file %@: %s\n",
					[e path], strerror([e errNo])];
		[OFApplication terminateWithStatus: 1];
	} @catch (OFReadFailedException *e) {
		[of_stderr writeFormat: @"Failed to read file %@: %s\n",
					path, strerror([e errNo])];
		[OFApplication terminateWithStatus: 1];
	} @catch (OFInvalidFormatException *e) {
		[of_stderr writeFormat: @"File %@ is not a valid archive!\n",
					path];
		[OFApplication terminateWithStatus: 1];
	}

	return archive;
}

- (void)listFilesInArchive: (OFZIPArchive*)archive
{
	OFEnumerator OF_GENERIC(OFZIPArchiveEntry*) *enumerator;
	OFZIPArchiveEntry *entry;

	enumerator = [[archive entries] objectEnumerator];

	while ((entry = [enumerator nextObject]) != nil) {
		void *pool = objc_autoreleasePoolPush();

		[of_stdout writeLine: [entry fileName]];

		if (_outputLevel >= 1) {
			OFString *date = [[entry modificationDate]
			    localDateStringWithFormat: @"%Y-%m-%d %H:%M:%S"];

			[of_stdout writeFormat:
			    @"\tCompressed: %" PRIu64 @" bytes\n"
			    @"\tUncompressed: %" PRIu64 @" bytes\n"
			    @"\tCRC32: %08X\n"
			    @"\tModification date: %@\n",
			    [entry compressedSize], [entry uncompressedSize],
			    [entry CRC32], date];

			if (_outputLevel >= 2) {
				uint16_t versionMadeBy = [entry versionMadeBy];

				[of_stdout writeFormat:
				    @"\tVersion made by: %@\n"
				    @"\tMinimum version needed: %@\n",
				    of_zip_archive_entry_version_to_string(
				    versionMadeBy),
				    of_zip_archive_entry_version_to_string(
				    [entry minVersionNeeded])];

				if ((versionMadeBy >> 8) ==
				    OF_ZIP_ARCHIVE_ENTRY_ATTR_COMPAT_UNIX) {
					uint32_t mode = [entry
					    versionSpecificAttributes] >> 16;
					[of_stdout writeFormat:
					    @"\tMode: %06o\n", mode];
				}
			}

			if (_outputLevel >= 3)
				[of_stdout writeFormat: @"\tExtra field: %@\n",
							[entry extraField]];

			if ([[entry fileComment] length] > 0)
				[of_stdout writeFormat: @"\tComment: %@\n",
							[entry fileComment]];
		}

		objc_autoreleasePoolPop(pool);
	}
}

- (void)extractFiles: (OFArray OF_GENERIC(OFString*)*)files
	 fromArchive: (OFZIPArchive*)archive
{
	OFEnumerator OF_GENERIC(OFZIPArchiveEntry*) *entryEnumerator;
	OFZIPArchiveEntry *entry;
	bool all;
	OFMutableSet OF_GENERIC(OFString*) *missing;

	all = ([files count] == 0);
	missing = [OFMutableSet setWithArray: files];

	entryEnumerator = [[archive entries] objectEnumerator];
	while ((entry = [entryEnumerator nextObject]) != nil) {
		void *pool = objc_autoreleasePoolPush();
		OFString *fileName = [entry fileName];
		OFString *outFileName = [fileName stringByStandardizingPath];
		OFArray OF_GENERIC(OFString*) *pathComponents;
		OFEnumerator OF_GENERIC(OFString*) *componentEnumerator;
		OFString *component, *directory;
		OFStream *stream;
		OFFile *output;
		char buffer[BUFFER_SIZE];
		uint64_t written = 0, size = [entry uncompressedSize];
		int_fast8_t percent = -1, newPercent;

		if (!all && ![files containsObject: fileName])
			continue;

		[missing removeObject: fileName];

#ifndef _WIN32
		if ([outFileName hasPrefix: @"/"]) {
#else
		if ([outFileName hasPrefix: @"/"] ||
		    [outFileName containsString: @":"]) {
#endif
			[of_stderr writeFormat: @"Refusing to extract %@!\n",
						fileName];

			_exitStatus = 1;
			goto outer_loop_end;
		}

		pathComponents = [outFileName pathComponents];
		componentEnumerator = [pathComponents objectEnumerator];
		while ((component = [componentEnumerator nextObject]) != nil) {
			if ([component isEqual: OF_PATH_PARENT_DIRECTORY]) {
				[of_stderr writeFormat:
				    @"Refusing to extract %@!\n", fileName];

				_exitStatus = 1;
				goto outer_loop_end;
			}
		}
		outFileName = [OFString pathWithComponents: pathComponents];

		if (_outputLevel >= 0)
			[of_stdout writeFormat: @"Extracting %@...", fileName];

		if ([fileName hasSuffix: @"/"]) {
			[OFFile createDirectoryAtPath: outFileName
					createParents: true];
			setPermissions(outFileName, entry);

			if (_outputLevel >= 0)
				[of_stdout writeLine: @" done"];

			goto outer_loop_end;
		}

		directory = [outFileName stringByDeletingLastPathComponent];
		if (![OFFile directoryExistsAtPath: directory])
			[OFFile createDirectoryAtPath: directory
					createParents: true];

		if ([OFFile fileExistsAtPath: outFileName] && _override != 1) {
			OFString *line;

			if (_override == -1) {
				if (_outputLevel >= 0)
					[of_stdout writeLine: @" skipped"];

				goto outer_loop_end;
			}

			do {
				[of_stderr writeFormat:
				    @"\rOverride %@? [ynAN?] ", fileName];

				line = [of_stdin readLine];

				if ([line isEqual: @"?"])
					[of_stderr writeString: @" y: yes\n"
								@" n: no\n"
								@" A: always\n"
								@" N: never\n"];
			} while (![line isEqual: @"y"] &&
			    ![line isEqual: @"n"] && ![line isEqual: @"N"] &&
			    ![line isEqual: @"A"]);

			if ([line isEqual: @"A"])
				_override = 1;
			else if ([line isEqual: @"N"])
				_override = -1;

			if ([line isEqual: @"n"] || [line isEqual: @"N"]) {
				[of_stdout writeFormat: @"Skipping %@...\n",
							fileName];

				goto outer_loop_end;
			}

			[of_stdout writeFormat: @"Extracting %@...", fileName];
		}

		stream = [archive streamForReadingFile: fileName];
		output = [OFFile fileWithPath: outFileName
					 mode: @"wb"];
		setPermissions(outFileName, entry);

		while (![stream isAtEndOfStream]) {
			size_t length;

			@try {
				length = [stream readIntoBuffer: buffer
							 length: BUFFER_SIZE];
			} @catch (OFReadFailedException *e) {
				[of_stderr writeFormat:
				    @"\rFailed to read file %@: %s\n",
				    fileName, strerror([e errNo])];

				_exitStatus = 1;
				goto outer_loop_end;
			}

			@try {
				[output writeBuffer: buffer
					     length: length];
			} @catch (OFWriteFailedException *e) {
				[of_stderr writeFormat:
				    @"\rFailed to write file %@: %s\n",
				    fileName, strerror([e errNo])];

				_exitStatus = 1;
				goto outer_loop_end;
			}

			written += length;
			newPercent = (written == size
			    ? 100 : (int_fast8_t)(written * 100 / size));

			if (_outputLevel >= 0 && percent != newPercent) {
				percent = newPercent;

				[of_stdout writeFormat:
				    @"\rExtracting %@... %3u%%",
				    fileName, percent];
			}
		}

		if (_outputLevel >= 0)
			[of_stdout writeFormat: @"\rExtracting %@... done\n",
						fileName];

outer_loop_end:
		objc_autoreleasePoolPop(pool);
	}

	if ([missing count] > 0) {
		OFEnumerator OF_GENERIC(OFString*) *enumerator;
		OFString *file;

		enumerator = [missing objectEnumerator];
		while ((file = [enumerator nextObject]) != nil)
			[of_stderr writeFormat:
			    @"File %@ is not in the archive!\n", file];

		_exitStatus = 1;
	}
}
@end
