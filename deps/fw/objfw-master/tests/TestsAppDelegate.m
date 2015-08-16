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

#import "ObjFW.h"

#import "TestsAppDelegate.h"

#if defined(STDOUT) && (defined(_WIN32) || defined(__DJGPP__))
# undef STDOUT
# define STDOUT_SIMPLE
#endif

#ifdef _PSP
# include <pspmoduleinfo.h>
# include <pspkernel.h>
# include <pspdebug.h>
# include <pspctrl.h>
PSP_MODULE_INFO("ObjFW Tests", 0, 0, 0);
#endif

#ifdef __wii__
# define BOOL OGC_BOOL
# define asm __asm__
# include <gccore.h>
# include <wiiuse/wpad.h>
# undef BOOL
# undef asm
#endif

#ifdef OF_NINTENDO_DS
# define asm __asm__
# include <nds.h>
# undef asm
#endif

enum {
	NO_COLOR,
	RED,
	GREEN,
	YELLOW
};

#ifdef _PSP
static int
exit_cb(int arg1, int arg2, void *arg)
{
	sceKernelExitGame();

	return 0;
}

static int
callback_thread(SceSize args, void *argp)
{
	sceKernelRegisterExitCallback(
	    sceKernelCreateCallback("Exit Callback", exit_cb, NULL));
	sceKernelSleepThreadCB();

	return 0;
}
#endif

int
main(int argc, char *argv[])
{
#ifdef _PSP
	int tid;
#endif

#if defined(OF_OBJFW_RUNTIME) && !defined(_WIN32)
	/* This does not work on Win32 if ObjFW is built as a DLL */
	atexit(objc_exit);
#endif

	/* We need deterministic hashes for tests */
	of_hash_seed = 0;

#ifdef __wii__
	GXRModeObj *rmode;
	void *xfb;

	VIDEO_Init();
	WPAD_Init();

	rmode = VIDEO_GetPreferredMode(NULL);
	xfb = MEM_K0_TO_K1(SYS_AllocateFramebuffer(rmode));
	VIDEO_Configure(rmode);
	VIDEO_SetNextFramebuffer(xfb);
	VIDEO_SetBlack(FALSE);
	VIDEO_Flush();

	VIDEO_WaitVSync();
	if (rmode->viTVMode & VI_NON_INTERLACE)
		VIDEO_WaitVSync();

	CON_InitEx(rmode, 10, 20, rmode->fbWidth - 10, rmode->xfbHeight - 20);
	VIDEO_ClearFrameBuffer(rmode, xfb, COLOR_BLACK);
#endif

#ifdef _PSP
	pspDebugScreenInit();

	sceCtrlSetSamplingCycle(0);
	sceCtrlSetSamplingMode(PSP_CTRL_MODE_DIGITAL);

	if ((tid = sceKernelCreateThread("update_thread", callback_thread,
	    0x11, 0xFA0, 0, 0)) >= 0)
		sceKernelStartThread(tid, 0, 0);
#endif

#ifdef OF_NINTENDO_DS
	consoleDemoInit();
#endif

#if defined(__wii__) || defined(_PSP) || defined(OF_NINTENDO_DS)
	@try {
		return of_application_main(&argc, &argv,
		    [TestsAppDelegate class]);
	} @catch (id e) {
		TestsAppDelegate *delegate =
		    [[OFApplication sharedApplication] delegate];
		OFString *string = [OFString stringWithFormat:
		    @"\nRuntime error: Unhandled exception:\n%@\n", e];
		OFString *backtrace = [OFString stringWithFormat:
		    @"\nBacktrace:\n  %@\n\n",
		    [[e backtrace] componentsJoinedByString: @"\n  "]];

		[delegate outputString: string
			       inColor: RED];
		[delegate outputString: backtrace
			       inColor: RED];
# if defined(__wii__)
		[delegate outputString: @"Press home button to exit!\n"
			       inColor: NO_COLOR];
		for (;;) {
			WPAD_ScanPads();

			if (WPAD_ButtonsDown(0) & WPAD_BUTTON_HOME)
				[OFApplication terminateWithStatus: 1];

			VIDEO_WaitVSync();
		}
# elif defined(_PSP)
		sceKernelSleepThreadCB();
# elif defined(OF_NINTENDO_DS)
		[delegate outputString: @"Press start button to exit!"
			       inColor: NO_COLOR];
		for (;;) {
			swiWaitForVBlank();
			scanKeys();
			if (keysDown() & KEY_START)
				[OFApplication terminateWithStatus: 1];
		}
# else
		abort();
# endif
	}
#else
	return of_application_main(&argc, &argv, [TestsAppDelegate class]);
#endif
}

@implementation TestsAppDelegate
- (void)outputString: (OFString*)str
	     inColor: (int)color
{
#if defined(_PSP)
	char i, space = ' ';
	int y = pspDebugScreenGetY();

	pspDebugScreenSetXY(0, y);
	for (i = 0; i < 68; i++)
		pspDebugScreenPrintData(&space, 1);

	switch (color) {
	case NO_COLOR:
		pspDebugScreenSetTextColor(0xFFFFFF);
		break;
	case RED:
		pspDebugScreenSetTextColor(0x0000FF);
		break;
	case GREEN:
		pspDebugScreenSetTextColor(0x00FF00);
		break;
	case YELLOW:
		pspDebugScreenSetTextColor(0x00FFFF);
		break;
	}

	pspDebugScreenSetXY(0, y);
	pspDebugScreenPrintData([str UTF8String], [str UTF8StringLength]);
#elif defined(STDOUT)
	switch (color) {
	case NO_COLOR:
		[of_stdout writeString: @"\r\033[K"];
# if defined(__wii__) || defined(OF_NINTENDO_DS)
		[of_stdout writeString: @"\033[37m"];
# endif
		break;
	case RED:
		[of_stdout writeString: @"\r\033[K\033[31;1m"];
		break;
	case GREEN:
		[of_stdout writeString: @"\r\033[K\033[32;1m"];
		break;
	case YELLOW:
		[of_stdout writeString: @"\r\033[K\033[33;1m"];
		break;
	}

	[of_stdout writeString: str];
	[of_stdout writeString: @"\033[m"];
#elif defined(STDOUT_SIMPLE)
	[of_stdout writeString: str];
#else
# error No output method!
#endif
}

- (void)outputTesting: (OFString*)test
	     inModule: (OFString*)module
{
	OFAutoreleasePool *pool = [[OFAutoreleasePool alloc] init];
#ifndef STDOUT_SIMPLE
	[self outputString: [OFString stringWithFormat: @"[%@] %@: testing...",
							module, test]
		   inColor: YELLOW];
#else
	[self outputString: [OFString stringWithFormat: @"[%@] %@: ",
							module, test]
		   inColor: YELLOW];
#endif
	[pool release];
}

- (void)outputSuccess: (OFString*)test
	     inModule: (OFString*)module
{
#ifndef STDOUT_SIMPLE
	OFAutoreleasePool *pool = [[OFAutoreleasePool alloc] init];
	[self outputString: [OFString stringWithFormat: @"[%@] %@: ok\n",
							module, test]
		   inColor: GREEN];
	[pool release];
#else
	[self outputString: @"ok\n"
		   inColor: GREEN];
#endif
}

- (void)outputFailure: (OFString*)test
	     inModule: (OFString*)module
{
#ifndef STDOUT_SIMPLE
	OFAutoreleasePool *pool = [[OFAutoreleasePool alloc] init];
	[self outputString: [OFString stringWithFormat: @"[%@] %@: failed\n",
							module, test]
		   inColor: RED];
	[pool release];

# ifdef __wii__
	[self outputString: @"Press A to continue!\n"
		   inColor: NO_COLOR];
	for (;;) {
		WPAD_ScanPads();

		if (WPAD_ButtonsDown(0) & WPAD_BUTTON_A)
			return;

		VIDEO_WaitVSync();
	}
# endif
# ifdef _PSP
	[self outputString: @"Press X to continue!\n"
		   inColor: NO_COLOR];
	for (;;) {
		SceCtrlData pad;

		sceCtrlReadBufferPositive(&pad, 1);
		if (pad.Buttons & PSP_CTRL_CROSS) {
			for (;;) {
				sceCtrlReadBufferPositive(&pad, 1);
				if (!(pad.Buttons & PSP_CTRL_CROSS))
				    return;
			}
		}
	}
# endif
# ifdef OF_NINTENDO_DS
	[self outputString: @"Press A to continue!"
		   inColor: NO_COLOR];
	for (;;) {
		swiWaitForVBlank();
		scanKeys();
		if (keysDown() & KEY_A)
			break;
	}
# endif
#else
	[self outputString: @"failed\n"
		   inColor: RED];
#endif
}

- (void)applicationDidFinishLaunching
{
#if defined(__wii__) && defined(OF_HAVE_FILES)
	[OFFile changeCurrentDirectoryPath: @"/apps/objfw-tests"];
#endif

	[self objectTests];
#ifdef OF_HAVE_BLOCKS
	[self blockTests];
#endif
	[self stringTests];
	[self dataArrayTests];
	[self arrayTests];
	[self dictionaryTests];
	[self listTests];
	[self setTests];
	[self dateTests];
	[self numberTests];
	[self streamTests];
#ifdef OF_HAVE_FILES
	[self MD5HashTests];
	[self RIPEMD160HashTests];
	[self SHA1HashTests];
	[self SHA224HashTests];
	[self SHA256HashTests];
	[self SHA384HashTests];
	[self SHA512HashTests];
	[self INIFileTests];
#endif
#ifdef OF_HAVE_SOCKETS
	[self TCPSocketTests];
	[self UDPSocketTests];
	[self kernelEventObserverTests];
#endif
#ifdef OF_HAVE_THREADS
	[self threadTests];
#endif
	[self URLTests];
#if defined(OF_HAVE_SOCKETS) && defined(OF_HAVE_THREADS)
	[self HTTPClientTests];
#endif
	[self XMLParserTests];
	[self XMLNodeTests];
	[self XMLElementBuilderTests];
#ifdef OF_HAVE_FILES
	[self serializationTests];
#endif
	[self JSONTests];
#ifdef OF_HAVE_PLUGINS
	[self pluginTests];
#endif
	[self forwardingTests];
#ifdef OF_HAVE_PROPERTIES
	[self propertiesTests];
#endif

#if defined(__wii__)
	[self outputString: @"Press home button to exit!\n"
		   inColor: NO_COLOR];
	for (;;) {
		WPAD_ScanPads();

		if (WPAD_ButtonsDown(0) & WPAD_BUTTON_HOME)
			[OFApplication terminateWithStatus: _fails];

		VIDEO_WaitVSync();
	}
#elif defined(_PSP)
	[self outputString: [OFString stringWithFormat: @"%d tests failed!",
							_fails]
		   inColor: NO_COLOR];
	sceKernelSleepThreadCB();
#elif defined(OF_NINTENDO_DS)
	[self outputString: @"Press start button to exit!"
		   inColor: NO_COLOR];
	for (;;) {
		swiWaitForVBlank();
		scanKeys();
		if (keysDown() & KEY_START)
			[OFApplication terminateWithStatus: _fails];
	}
#else
	[OFApplication terminateWithStatus: _fails];
#endif
}
@end
