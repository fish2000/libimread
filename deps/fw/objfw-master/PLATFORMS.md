Platforms
=========

ObjFW is known to work on the following platforms, but should run on many
others as well.


Android
-------

  * OS Versions: 4.0.4, 4.1.2
  * Architectures: ARMv6, ARMv7
  * Compilers: Clang 3.3
  * Runtimes: ObjFW
  * Limitations: Does not work as a shared library


Bare metal ARM Cortex-M4
------------------------

  * Architectures: ARMv7E-M
  * Compilers: Clang 3.5
  * Runtimes: ObjFW
  * Limitations: No threads, no sockets, no files
  * Note: Bootloader, libc (newlib) and possibly external RAM required


DOS
---

  * OS Versions: Windows XP DOS Emulation, DOSBox
  * Architectures: x86
  * Compilers: DJGPP GCC 4.7.3 (djdev204)
  * Runtimes: ObjFW


DragonFlyBSD
------------

  * OS Versions: 3.0, 3.3-DEVELOPMENT
  * Architectures: x86, x86_64
  * Compilers: GCC 4.4.7
  * Runtimes: ObjFW


FreeBSD
-------

  * OS Versions: 9.1-rc3, 10.0
  * Architectures: x86_64
  * Compilers: Clang 3.1, Clang 3.3
  * Runtimes: ObjFW


Haiku
-----

  * OS version: r1-alpha4
  * Architectures: x86
  * Compilers: Clang 3.2, GCC 4.6.3
  * Runtimes: ObjFW


iOS
---

  * Architectures: ARMv7, ARM64
  * Compilers: GCC 4.2.1
  * Runtimes: Apple


Linux
-----

  * Architectures: Alpha, ARMv6, ARM64, m68k, MIPS (O32), PPC, SH4, x86, x86_64
  * Compilers: Clang 3.0-3.6, GCC 4.2, GCC 4.6-4.8
  * Runtimes: ObjFW


Mac OS X
--------

  * OS Versions: 10.5, 10.7-10.10
  * Architectures: PPC, PPC64, x86, x86_64
  * Compilers: Clang 3.1-3.7, LLVM GCC 4.2.1
  * Runtimes: Apple, ObjFW


NetBSD
------

  * OS Versions: 5.1-6.1
  * Architectures: SPARC, SPARC64, x86, x86_64
  * Compilers: Clang 3.0-3.2, GCC 4.1.3 & 4.5.3
  * Runtimes: ObjFW


Nintendo DS
-----------

  * Architectures: ARM (EABI)
  * Compilers: GCC 4.8.2 (devkitARM release 42)
  * Runtimes: ObjFW
  * Limitations: No threads, no sockets
  * Note: File support requires an argv-compatible launcher (such as HBMenu)


OpenBSD
-------

  * OS Versions: 5.2-5.7
  * Architectures: MIPS64, PPC, SPARC64, x86_64
  * Compilers: GCC 4.2.1, Clang 3.5
  * Runtimes: ObjFW


PlayStation Portable
--------------------

  * OS Versions: 5.00 M33-4
  * Architectures: MIPS (EABI)
  * Compiler: GCC 4.6.2 (devkitPSP release 16)
  * Runtimes: ObjFW
  * Limitations: No threads, no sockets


QNX
---

  * OS Versions: 6.5.0
  * Architectures: x86
  * Compilers: GCC 4.6.1
  * Runtimes: ObjFW


Solaris
-------

  * OS Versions: OpenIndiana 2015.03
  * Architectures: x86, x86_64
  * Compilers: Clang 3.4.2, GCC 4.8.3
  * Runtimes: ObjFW


Wii
---

  * OS Versions: 4.3E / Homebrew Channel 1.1.0
  * Architectures: PPC
  * Compilers: GCC 4.6.3 (devkitPPC release 26)
  * Runtimes: ObjFW
  * Limitations: No threads


Windows
-------

  * OS Versions: XP (x86), 7 (x64), 8 (x64), 8.1 (x64), Wine (x86 & x64)
  * Architectures: x86, x86_64
  * Compilers: TDM GCC 4.6.1-dw2, TDM GCC 4.7.1-dw2, MinGW-w64 GCC 4.8.2 DWARF,
               MinGW-w64 GCC 4.8.2 SEH, MinGW-w64 GCC 4.8.2 SjLj
  * Runtimes: ObjFW


Others
------

Basically, it should run on any POSIX system to which GCC 4 or a recent Clang
version has been ported. If not, please send an e-mail with a bug report.

If you successfully ran ObjFW on a platform not listed here, please send an
e-mail to js@webkeks.org so it can be added here!

If you have a platform on which ObjFW does not work, please contact me as well!


Forwarding
==========

As forwarding needs hand-written assembly for each combination of CPU
architecture, executable format and calling convention, it is only available
for the following platforms (except resolveClassMethod: and
resolveInstanceMethod:, which are always available):

  * ARM (EABI/ELF, Apple/Mach-O)
  * ARM64 (Apple/Mach-O)
  * MIPS (O32/ELF, EABI/ELF)
  * PPC (SysV/ELF, EABI/ELF, Apple/Mach-O)
  * x86 (SysV/ELF, Apple/Mach-O, Win32/PE)
  * x86_64 (SysV/ELF, Apple/Mach-O, Win64/PE)

Apple means both, the Apple ABI and runtime.
