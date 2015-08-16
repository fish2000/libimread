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

#define __NO_EXT_QNX

#include "config.h"

/* Work around a bug with Clang + glibc */
#ifdef __clang__
# define _HAVE_STRING_ARCH_strcmp
#endif

#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Work around __block being used by glibc */
#ifdef __GLIBC__
# undef __USE_XOPEN
#endif

#include <unistd.h>

#include <dirent.h>
#include <fcntl.h>

#ifdef HAVE_PWD_H
# include <pwd.h>
#endif
#ifdef HAVE_GRP_H
# include <grp.h>
#endif

#ifdef __wii__
# define BOOL OGC_BOOL
# include <fat.h>
# undef BOOL
#endif

#ifdef OF_NINTENDO_DS
# include <filesystem.h>
#endif

#import "OFFile.h"
#import "OFString.h"
#import "OFArray.h"
#ifdef OF_HAVE_THREADS
# import "threading.h"
#endif
#import "OFDate.h"
#import "OFSystemInfo.h"

#import "OFChangeCurrentDirectoryPathFailedException.h"
#import "OFChangeOwnerFailedException.h"
#import "OFChangePermissionsFailedException.h"
#import "OFCopyItemFailedException.h"
#import "OFCreateDirectoryFailedException.h"
#import "OFCreateSymbolicLinkFailedException.h"
#import "OFInitializationFailedException.h"
#import "OFInvalidArgumentException.h"
#import "OFLinkFailedException.h"
#import "OFLockFailedException.h"
#import "OFMoveItemFailedException.h"
#import "OFOpenItemFailedException.h"
#import "OFOutOfMemoryException.h"
#import "OFOutOfRangeException.h"
#import "OFReadFailedException.h"
#import "OFRemoveItemFailedException.h"
#import "OFStatItemFailedException.h"
#import "OFSeekFailedException.h"
#import "OFUnlockFailedException.h"
#import "OFWriteFailedException.h"

#ifdef _WIN32
# include <windows.h>
# include <direct.h>
#endif

#ifndef O_BINARY
# define O_BINARY 0
#endif
#ifndef O_CLOEXEC
# define O_CLOEXEC 0
#endif
#ifndef O_EXLOCK
# define O_EXLOCK 0
#endif

#ifndef S_IRGRP
# define S_IRGRP 0
#endif
#ifndef S_IROTH
# define S_IROTH 0
#endif
#ifndef S_IWGRP
# define S_IWGRP 0
#endif
#ifndef S_IWOTH
# define S_IWOTH 0
#endif

#define DEFAULT_MODE S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH
#define DIR_MODE DEFAULT_MODE | S_IXUSR | S_IXGRP | S_IXOTH

#if defined(OF_HAVE_CHOWN) && defined(OF_HAVE_THREADS)
static of_mutex_t mutex;
#endif
#if !defined(HAVE_READDIR_R) && !defined(_WIN32) && defined(OF_HAVE_THREADS)
static of_mutex_t mutex;
#endif

int
of_stat(OFString *path, of_stat_t *buffer)
{
#if defined(_WIN32)
	return _wstat64([path UTF16String], buffer);
#elif defined(OF_HAVE_OFF64_T)
	return stat64([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]], buffer);
#else
	return stat([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]], buffer);
#endif
}

int
of_lstat(OFString *path, of_stat_t *buffer)
{
#if defined(_WIN32)
	return _wstat64([path UTF16String], buffer);
#elif defined(HAVE_LSTAT)
# ifdef OF_HAVE_OFF64_T
	return lstat64([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]], buffer);
# else
	return lstat([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]], buffer);
# endif
#else
# ifdef OF_HAVE_OFF64_T
	return stat64([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]], buffer);
# else
	return stat([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]], buffer);
# endif
#endif
}

static int
parseMode(const char *mode)
{
	if (strcmp(mode, "r") == 0)
		return O_RDONLY;
	if (strcmp(mode, "w") == 0)
		return O_WRONLY | O_CREAT | O_TRUNC;
	if (strcmp(mode, "wx") == 0)
		return O_WRONLY | O_CREAT | O_EXCL | O_EXLOCK;
	if (strcmp(mode, "a") == 0)
		return O_WRONLY | O_CREAT | O_APPEND;
	if (strcmp(mode, "rb") == 0)
		return O_RDONLY | O_BINARY;
	if (strcmp(mode, "wb") == 0)
		return O_WRONLY | O_CREAT | O_TRUNC | O_BINARY;
	if (strcmp(mode, "wbx") == 0)
		return O_WRONLY | O_CREAT | O_EXCL | O_EXLOCK;
	if (strcmp(mode, "ab") == 0)
		return O_WRONLY | O_CREAT | O_APPEND | O_BINARY;
	if (strcmp(mode, "r+") == 0)
		return O_RDWR;
	if (strcmp(mode, "w+") == 0)
		return O_RDWR | O_CREAT | O_TRUNC;
	if (strcmp(mode, "a+") == 0)
		return O_RDWR | O_CREAT | O_APPEND;
	if (strcmp(mode, "r+b") == 0 || strcmp(mode, "rb+") == 0)
		return O_RDWR | O_BINARY;
	if (strcmp(mode, "w+b") == 0 || strcmp(mode, "wb+") == 0)
		return O_RDWR | O_CREAT | O_TRUNC | O_BINARY;
	if (strcmp(mode, "w+bx") == 0 || strcmp(mode, "wb+x") == 0)
		return O_RDWR | O_CREAT | O_EXCL | O_EXLOCK;
	if (strcmp(mode, "ab+") == 0 || strcmp(mode, "a+b") == 0)
		return O_RDWR | O_CREAT | O_APPEND | O_BINARY;

	return -1;
}

@implementation OFFile
+ (void)initialize
{
	if (self != [OFFile class])
		return;

#if defined(OF_HAVE_CHOWN) && defined(OF_HAVE_THREADS)
	if (!of_mutex_new(&mutex))
		@throw [OFInitializationFailedException
		    exceptionWithClass: self];
#endif

#if !defined(HAVE_READDIR_R) && !defined(_WIN32) && defined(OF_HAVE_THREADS)
	if (!of_mutex_new(&mutex))
		@throw [OFInitializationFailedException
		    exceptionWithClass: self];
#endif

#ifdef __wii__
	if (!fatInitDefault())
		@throw [OFInitializationFailedException
		    exceptionWithClass: self];
#endif

#ifdef OF_NINTENDO_DS
	if (!nitroFSInit(NULL))
		@throw [OFInitializationFailedException
		    exceptionWithClass: self];
#endif
}

+ (instancetype)fileWithPath: (OFString*)path
			mode: (OFString*)mode
{
	return [[[self alloc] initWithPath: path
				      mode: mode] autorelease];
}

+ (instancetype)fileWithFileDescriptor: (int)filedescriptor
{
	return [[[self alloc]
	    initWithFileDescriptor: filedescriptor] autorelease];
}

+ (OFString*)currentDirectoryPath
{
	OFString *ret;
#ifndef _WIN32
	char *buffer = getcwd(NULL, 0);
#else
	wchar_t *buffer = _wgetcwd(NULL, 0);
#endif

	@try {
#ifndef _WIN32
		ret = [OFString
		    stringWithCString: buffer
			     encoding: [OFSystemInfo native8BitEncoding]];
#else
		ret = [OFString stringWithUTF16String: buffer];
#endif
	} @finally {
		free(buffer);
	}

	return ret;
}

+ (bool)fileExistsAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_stat(path, &s) == -1)
		return false;

	if (S_ISREG(s.st_mode))
		return true;

	return false;
}

+ (bool)directoryExistsAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_stat(path, &s) == -1)
		return false;

	if (S_ISDIR(s.st_mode))
		return true;

	return false;
}

#ifdef OF_HAVE_SYMLINK
+ (bool)symbolicLinkExistsAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_lstat(path, &s) == -1)
		return false;

	if (S_ISLNK(s.st_mode))
		return true;

	return false;
}
#endif

+ (void)createDirectoryAtPath: (OFString*)path
{
	if (path == nil)
		@throw [OFInvalidArgumentException exception];

#ifndef _WIN32
	if (mkdir([path cStringWithEncoding: [OFSystemInfo native8BitEncoding]],
	    DIR_MODE) != 0)
#else
	if (_wmkdir([path UTF16String]) != 0)
#endif
		@throw [OFCreateDirectoryFailedException
		    exceptionWithPath: path
				errNo: errno];
}

+ (void)createDirectoryAtPath: (OFString*)path
		createParents: (bool)createParents
{
	void *pool;
	OFArray *pathComponents;
	OFString *currentPath = nil, *component;
	OFEnumerator *enumerator;

	if (!createParents) {
		[OFFile createDirectoryAtPath: path];
		return;
	}

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	pool = objc_autoreleasePoolPush();

	pathComponents = [path pathComponents];
	enumerator = [pathComponents objectEnumerator];
	while ((component = [enumerator nextObject]) != nil) {
		void *pool2 = objc_autoreleasePoolPush();

		if (currentPath != nil)
			currentPath = [currentPath
			    stringByAppendingPathComponent: component];
		else
			currentPath = component;

		if ([currentPath length] > 0 &&
		    ![OFFile directoryExistsAtPath: currentPath])
			[OFFile createDirectoryAtPath: currentPath];

		[currentPath retain];

		objc_autoreleasePoolPop(pool2);

		[currentPath autorelease];
	}

	objc_autoreleasePoolPop(pool);
}

+ (OFArray*)contentsOfDirectoryAtPath: (OFString*)path
{
	OFMutableArray *files;
#ifndef _WIN32
	of_string_encoding_t encoding;
#endif

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	files = [OFMutableArray array];

#ifndef _WIN32
	DIR *dir;

	encoding = [OFSystemInfo native8BitEncoding];

	if ((dir = opendir([path cStringWithEncoding: encoding])) == NULL)
		@throw [OFOpenItemFailedException exceptionWithPath: path
							      errNo: errno];

# if !defined(HAVE_READDIR_R) && defined(OF_HAVE_THREADS)
	if (!of_mutex_lock(&mutex))
		@throw [OFLockFailedException exception];
# endif

	@try {
		for (;;) {
			struct dirent *dirent;
# ifdef HAVE_READDIR_R
			struct dirent buffer;
# endif
			void *pool;
			OFString *file;

# ifdef HAVE_READDIR_R
			if (readdir_r(dir, &buffer, &dirent) != 0)
				@throw [OFReadFailedException
				    exceptionWithObject: self
					requestedLength: 0
						  errNo: errno];

			if (dirent == NULL)
				break;
# else
			if ((dirent = readdir(dir)) == NULL) {
				if (errno == 0)
					break;
				else
					@throw [OFReadFailedException
					    exceptionWithObject: self
						requestedLength: 0
							  errNo: errno];
			}
# endif

			if (strcmp(dirent->d_name, ".") == 0 ||
			    strcmp(dirent->d_name, "..") == 0)
				continue;

			pool = objc_autoreleasePoolPush();

			file = [OFString stringWithCString: dirent->d_name
						  encoding: encoding];
			[files addObject: file];

			objc_autoreleasePoolPop(pool);
		}
	} @finally {
		closedir(dir);
# if !defined(HAVE_READDIR_R) && defined(OF_HAVE_THREADS)
		if (!of_mutex_unlock(&mutex))
			@throw [OFUnlockFailedException exception];
# endif
	}
#else
	void *pool = objc_autoreleasePoolPush();
	HANDLE handle;
	WIN32_FIND_DATAW fd;

	path = [path stringByAppendingString: @"\\*"];

	if ((handle = FindFirstFileW([path UTF16String],
	    &fd)) == INVALID_HANDLE_VALUE) {
		int errNo = 0;

		if (GetLastError() == ERROR_FILE_NOT_FOUND)
			errNo = ENOENT;

		@throw [OFOpenItemFailedException exceptionWithPath: path
							      errNo: errNo];
	}

	@try {
		do {
			void *pool2 = objc_autoreleasePoolPush();
			OFString *file;

			if (!wcscmp(fd.cFileName, L".") ||
			    !wcscmp(fd.cFileName, L".."))
				continue;

			file = [OFString stringWithUTF16String: fd.cFileName];
			[files addObject: file];

			objc_autoreleasePoolPop(pool2);
		} while (FindNextFileW(handle, &fd));

		if (GetLastError() != ERROR_NO_MORE_FILES)
			@throw [OFReadFailedException exceptionWithObject: self
							  requestedLength: 0];
	} @finally {
		FindClose(handle);
	}

	objc_autoreleasePoolPop(pool);
#endif

	[files makeImmutable];

	return files;
}

+ (void)changeCurrentDirectoryPath: (OFString*)path
{
	if (path == nil)
		@throw [OFInvalidArgumentException exception];

#ifndef _WIN32
	if (chdir([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]]) != 0)
#else
	if (_wchdir([path UTF16String]) != 0)
#endif
		@throw [OFChangeCurrentDirectoryPathFailedException
		    exceptionWithPath: path
				errNo: errno];
}

+ (of_offset_t)sizeOfFileAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_stat(path, &s) != 0)
		@throw [OFStatItemFailedException exceptionWithPath: path
							      errNo: errno];

	return s.st_size;
}

+ (OFDate*)accessTimeOfItemAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_stat(path, &s) != 0)
		@throw [OFStatItemFailedException exceptionWithPath: path
							      errNo: errno];

	/* FIXME: We could be more precise on some OSes */
	return [OFDate dateWithTimeIntervalSince1970: s.st_atime];
}

+ (OFDate*)modificationTimeOfItemAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_stat(path, &s) != 0)
		@throw [OFStatItemFailedException exceptionWithPath: path
							      errNo: errno];

	/* FIXME: We could be more precise on some OSes */
	return [OFDate dateWithTimeIntervalSince1970: s.st_mtime];
}

+ (OFDate*)statusChangeTimeOfItemAtPath: (OFString*)path
{
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	if (of_stat(path, &s) != 0)
		@throw [OFStatItemFailedException exceptionWithPath: path
							      errNo: errno];

	/* FIXME: We could be more precise on some OSes */
	return [OFDate dateWithTimeIntervalSince1970: s.st_ctime];
}

#ifdef OF_HAVE_CHMOD
+ (void)changePermissionsOfItemAtPath: (OFString*)path
			  permissions: (mode_t)permissions
{
	if (path == nil)
		@throw [OFInvalidArgumentException exception];

# ifndef _WIN32
	if (chmod([path cStringWithEncoding: [OFSystemInfo native8BitEncoding]],
	    permissions) != 0)
# else
	if (_wchmod([path UTF16String], permissions) != 0)
# endif
		@throw [OFChangePermissionsFailedException
		    exceptionWithPath: path
			  permissions: permissions
				errNo: errno];
}
#endif

#ifdef OF_HAVE_CHOWN
+ (void)changeOwnerOfItemAtPath: (OFString*)path
			  owner: (OFString*)owner
			  group: (OFString*)group
{
	uid_t uid = -1;
	gid_t gid = -1;
	of_string_encoding_t encoding;

	if (path == nil || (owner == nil && group == nil))
		@throw [OFInvalidArgumentException exception];

	encoding = [OFSystemInfo native8BitEncoding];

# ifdef OF_HAVE_THREADS
	if (!of_mutex_lock(&mutex))
		@throw [OFLockFailedException exception];

	@try {
# endif
		if (owner != nil) {
			struct passwd *passwd;

			if ((passwd = getpwnam([owner
			    cStringWithEncoding: encoding])) == NULL)
				@throw [OFChangeOwnerFailedException
				    exceptionWithPath: path
						owner: owner
						group: group
						errNo: errno];

			uid = passwd->pw_uid;
		}

		if (group != nil) {
			struct group *group_;

			if ((group_ = getgrnam([group
			    cStringWithEncoding: encoding])) == NULL)
				@throw [OFChangeOwnerFailedException
				    exceptionWithPath: path
						owner: owner
						group: group
						errNo: errno];

			gid = group_->gr_gid;
		}
# ifdef OF_HAVE_THREADS
	} @finally {
		if (!of_mutex_unlock(&mutex))
			@throw [OFUnlockFailedException exception];
	}
# endif

	if (chown([path cStringWithEncoding: encoding], uid, gid) != 0)
		@throw [OFChangeOwnerFailedException exceptionWithPath: path
								 owner: owner
								 group: group
								 errNo: errno];
}
#endif

+ (void)copyItemAtPath: (OFString*)source
		toPath: (OFString*)destination
{
	void *pool;
	of_stat_t s;

	if (source == nil || destination == nil)
		@throw [OFInvalidArgumentException exception];

	pool = objc_autoreleasePoolPush();

	if (of_lstat(destination, &s) == 0)
		@throw [OFCopyItemFailedException
		    exceptionWithSourcePath: source
			    destinationPath: destination
				      errNo: EEXIST];

	if (of_lstat(source, &s) != 0)
		@throw [OFCopyItemFailedException
		    exceptionWithSourcePath: source
			    destinationPath: destination
				      errNo: errno];

	if (S_ISDIR(s.st_mode)) {
		OFArray *contents;
		OFEnumerator *enumerator;
		OFString *item;

		@try {
			[OFFile createDirectoryAtPath: destination];
#ifdef OF_HAVE_CHMOD
			[OFFile changePermissionsOfItemAtPath: destination
						  permissions: s.st_mode];
#endif

			contents = [OFFile contentsOfDirectoryAtPath: source];
		} @catch (id e) {
			/*
			 * Only convert exceptions to OFCopyItemFailedException
			 * that have an errNo property. This covers all I/O
			 * related exceptions from the operations used to copy
			 * an item, all others should be left as is.
			 */
			if ([e respondsToSelector: @selector(errNo)])
				@throw [OFCopyItemFailedException
				    exceptionWithSourcePath: source
					    destinationPath: destination
						      errNo: [e errNo]];

			@throw e;
		}

		enumerator = [contents objectEnumerator];
		while ((item = [enumerator nextObject]) != nil) {
			void *pool2 = objc_autoreleasePoolPush();
			OFString *sourcePath, *destinationPath;

			sourcePath =
			    [source stringByAppendingPathComponent: item];
			destinationPath =
			    [destination stringByAppendingPathComponent: item];

			[OFFile copyItemAtPath: sourcePath
					toPath: destinationPath];

			objc_autoreleasePoolPop(pool2);
		}
	} else if (S_ISREG(s.st_mode)) {
		size_t pageSize = [OFSystemInfo pageSize];
		OFFile *sourceFile = nil;
		OFFile *destinationFile = nil;
		char *buffer;

		if ((buffer = malloc(pageSize)) == NULL)
			@throw [OFOutOfMemoryException
			    exceptionWithRequestedSize: pageSize];

		@try {
			sourceFile = [OFFile fileWithPath: source
						     mode: @"rb"];
			destinationFile = [OFFile fileWithPath: destination
							  mode: @"wb"];

			while (![sourceFile isAtEndOfStream]) {
				size_t length;

				length = [sourceFile readIntoBuffer: buffer
							     length: pageSize];
				[destinationFile writeBuffer: buffer
						      length: length];
			}

#ifdef OF_HAVE_CHMOD
			[self changePermissionsOfItemAtPath: destination
						permissions: s.st_mode];
#endif
		} @catch (id e) {
			/*
			 * Only convert exceptions to OFCopyItemFailedException
			 * that have an errNo property. This covers all I/O
			 * related exceptions from the operations used to copy
			 * an item, all others should be left as is.
			 */
			if ([e respondsToSelector: @selector(errNo)])
				@throw [OFCopyItemFailedException
				    exceptionWithSourcePath: source
					    destinationPath: destination
						      errNo: [e errNo]];

			@throw e;
		} @finally {
			[sourceFile close];
			[destinationFile close];
			free(buffer);
		}
#ifdef OF_HAVE_SYMLINK
	} else if (S_ISLNK(s.st_mode)) {
		@try {
			source = [OFFile
			    destinationOfSymbolicLinkAtPath: source];

			[OFFile createSymbolicLinkAtPath: destination
				     withDestinationPath: source];
		} @catch (id e) {
			/*
			 * Only convert exceptions to OFCopyItemFailedException
			 * that have an errNo property. This covers all I/O
			 * related exceptions from the operations used to copy
			 * an item, all others should be left as is.
			 */
			if ([e respondsToSelector: @selector(errNo)])
				@throw [OFCopyItemFailedException
				    exceptionWithSourcePath: source
					    destinationPath: destination
						      errNo: [e errNo]];

			@throw e;
		}
#endif
	} else
		@throw [OFCopyItemFailedException
		    exceptionWithSourcePath: source
			    destinationPath: destination
				      errNo: ENOTSUP];

	objc_autoreleasePoolPop(pool);
}

+ (void)moveItemAtPath: (OFString*)source
		toPath: (OFString*)destination
{
	void *pool;
	of_stat_t s;
#ifndef _WIN32
	of_string_encoding_t encoding;
#endif

	if (source == nil || destination == nil)
		@throw [OFInvalidArgumentException exception];

	pool = objc_autoreleasePoolPush();

	if (of_lstat(destination, &s) == 0)
		@throw [OFCopyItemFailedException
		    exceptionWithSourcePath: source
			    destinationPath: destination
				      errNo: EEXIST];

#ifndef _WIN32
	encoding = [OFSystemInfo native8BitEncoding];

	if (rename([source cStringWithEncoding: encoding],
	    [destination cStringWithEncoding: encoding]) != 0) {
#else
	if (_wrename([source UTF16String], [destination UTF16String]) != 0) {
#endif
		if (errno != EXDEV)
			@throw [OFMoveItemFailedException
			    exceptionWithSourcePath: source
				    destinationPath: destination
					      errNo: errno];

		@try {
			[OFFile copyItemAtPath: source
					toPath: destination];
		} @catch (OFCopyItemFailedException *e) {
			[OFFile removeItemAtPath: destination];

			@throw [OFMoveItemFailedException
			    exceptionWithSourcePath: source
				    destinationPath: destination
					      errNo: [e errNo]];
		}

		@try {
			[OFFile removeItemAtPath: source];
		} @catch (OFRemoveItemFailedException *e) {
			@throw [OFMoveItemFailedException
			    exceptionWithSourcePath: source
				    destinationPath: destination
					      errNo: [e errNo]];
		}
	}

	objc_autoreleasePoolPop(pool);
}

+ (void)removeItemAtPath: (OFString*)path
{
	void *pool;
	of_stat_t s;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	pool = objc_autoreleasePoolPush();

	if (of_lstat(path, &s) != 0)
		@throw [OFRemoveItemFailedException exceptionWithPath: path
								errNo: errno];

	if (S_ISDIR(s.st_mode)) {
		OFArray *contents;
		OFEnumerator *enumerator;
		OFString *item;

		@try {
			contents = [OFFile contentsOfDirectoryAtPath: path];
		} @catch (id e) {
			/*
			 * Only convert exceptions to
			 * OFRemoveItemFailedException that have an errNo
			 * property. This covers all I/O related exceptions
			 * from the operations used to remove an item, all
			 * others should be left as is.
			 */
			if ([e respondsToSelector: @selector(errNo)])
				@throw [OFRemoveItemFailedException
				    exceptionWithPath: path
						errNo: [e errNo]];

			@throw e;
		}

		enumerator = [contents objectEnumerator];
		while ((item = [enumerator nextObject]) != nil) {
			void *pool2 = objc_autoreleasePoolPush();

			[OFFile removeItemAtPath:
			    [path stringByAppendingPathComponent: item]];

			objc_autoreleasePoolPop(pool2);
		}
	}

#ifndef _WIN32
	if (remove([path cStringWithEncoding:
	    [OFSystemInfo native8BitEncoding]]) != 0)
#else
	if (_wremove([path UTF16String]) != 0)
#endif
		@throw [OFRemoveItemFailedException exceptionWithPath: path
								errNo: errno];

	objc_autoreleasePoolPop(pool);
}

#ifdef OF_HAVE_LINK
+ (void)linkItemAtPath: (OFString*)source
		toPath: (OFString*)destination
{
	void *pool;
	of_string_encoding_t encoding;

	if (source == nil || destination == nil)
		@throw [OFInvalidArgumentException exception];

	pool = objc_autoreleasePoolPush();
	encoding = [OFSystemInfo native8BitEncoding];

	if (link([source cStringWithEncoding: encoding],
	    [destination cStringWithEncoding: encoding]) != 0)
		@throw [OFLinkFailedException
		    exceptionWithSourcePath: source
			    destinationPath: destination
				      errNo: errno];

	objc_autoreleasePoolPop(pool);
}
#endif

#ifdef OF_HAVE_SYMLINK
+ (void)createSymbolicLinkAtPath: (OFString*)destination
	     withDestinationPath: (OFString*)source
{
	void *pool;
	of_string_encoding_t encoding;

	if (source == nil || destination == nil)
		@throw [OFInvalidArgumentException exception];

	pool = objc_autoreleasePoolPush();
	encoding = [OFSystemInfo native8BitEncoding];

	if (symlink([source cStringWithEncoding: encoding],
	    [destination cStringWithEncoding: encoding]) != 0)
		@throw [OFCreateSymbolicLinkFailedException
		    exceptionWithSourcePath: source
			    destinationPath: destination
				      errNo: errno];

	objc_autoreleasePoolPop(pool);
}

+ (OFString*)destinationOfSymbolicLinkAtPath: (OFString*)path
{
	char destination[PATH_MAX];
	ssize_t length;
	of_string_encoding_t encoding;

	if (path == nil)
		@throw [OFInvalidArgumentException exception];

	encoding = [OFSystemInfo native8BitEncoding];
	length = readlink([path cStringWithEncoding: encoding],
	    destination, PATH_MAX);

	if (length < 0)
		@throw [OFStatItemFailedException exceptionWithPath: path
							      errNo: errno];

	return [OFString stringWithCString: destination
				  encoding: encoding
				    length: length];
}
#endif

- init
{
	OF_INVALID_INIT_METHOD
}

- initWithPath: (OFString*)path
	  mode: (OFString*)mode
{
	self = [super init];

	@try {
		int flags;

		if ((flags = parseMode([mode UTF8String])) == -1)
			@throw [OFInvalidArgumentException exception];

		flags |= O_CLOEXEC;

#if defined(_WIN32)
		if ((_fd = _wopen([path UTF16String], flags,
		    DEFAULT_MODE)) == -1)
#elif defined(OF_HAVE_OFF64_T)
		if ((_fd = open64([path cStringWithEncoding: [OFSystemInfo
		    native8BitEncoding]], flags, DEFAULT_MODE)) == -1)
#else
		if ((_fd = open([path cStringWithEncoding: [OFSystemInfo
		    native8BitEncoding]], flags, DEFAULT_MODE)) == -1)
#endif
			@throw [OFOpenItemFailedException
			    exceptionWithPath: path
					 mode: mode
					errNo: errno];
	} @catch (id e) {
		[self release];
		@throw e;
	}

	return self;
}

- initWithFileDescriptor: (int)fd
{
	self = [super init];

	_fd = fd;

	return self;
}

- (bool)lowlevelIsAtEndOfStream
{
	if (_fd == -1)
		return true;

	return _atEndOfStream;
}

- (size_t)lowlevelReadIntoBuffer: (void*)buffer
			  length: (size_t)length
{
	ssize_t ret;

	if (_fd == -1 || _atEndOfStream)
		@throw [OFReadFailedException exceptionWithObject: self
						  requestedLength: length];

#ifndef _WIN32
	if ((ret = read(_fd, buffer, length)) < 0)
		@throw [OFReadFailedException exceptionWithObject: self
						  requestedLength: length
							    errNo: errno];
#else
	if (length > UINT_MAX)
		@throw [OFOutOfRangeException exception];

	if ((ret = read(_fd, buffer, (unsigned int)length)) < 0)
		@throw [OFReadFailedException exceptionWithObject: self
						  requestedLength: length
							    errNo: errno];
#endif

	if (ret == 0)
		_atEndOfStream = true;

	return ret;
}

- (void)lowlevelWriteBuffer: (const void*)buffer
		     length: (size_t)length
{
	if (_fd == -1 || _atEndOfStream)
		@throw [OFWriteFailedException exceptionWithObject: self
						   requestedLength: length];

#ifndef _WIN32
	if (write(_fd, buffer, length) < length)
		@throw [OFWriteFailedException exceptionWithObject: self
						   requestedLength: length
							     errNo: errno];
#else
	if (length > UINT_MAX)
		@throw [OFOutOfRangeException exception];

	if (write(_fd, buffer, (unsigned int)length) < length)
		@throw [OFWriteFailedException exceptionWithObject: self
						   requestedLength: length
							     errNo: errno];
#endif
}

- (of_offset_t)lowlevelSeekToOffset: (of_offset_t)offset
			     whence: (int)whence
{
#if defined(_WIN32)
	of_offset_t ret = _lseeki64(_fd, offset, whence);
#elif defined(OF_HAVE_OFF64_T)
	of_offset_t ret = lseek64(_fd, offset, whence);
#else
	of_offset_t ret = lseek(_fd, offset, whence);
#endif

	if (ret == -1)
		@throw [OFSeekFailedException exceptionWithStream: self
							   offset: offset
							   whence: whence
							    errNo: errno];

	_atEndOfStream = false;

	return ret;
}

- (int)fileDescriptorForReading
{
	return _fd;
}

- (int)fileDescriptorForWriting
{
	return _fd;
}

- (void)close
{
	if (_fd != -1)
		close(_fd);

	_fd = -1;
}

- (void)dealloc
{
	if (_fd != -1)
		close(_fd);

	[super dealloc];
}
@end
