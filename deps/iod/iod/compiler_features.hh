#pragma once

/// Compatibility with non-clang compilers.
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

/// Compatibility with non-clang compilers.
#ifndef __has_feature
#define __has_feature(x) 0
#endif

/// Compatibility with pre-3.0 compilers.
#ifndef __has_extension
#define __has_extension __has_feature
#endif

#if __has_extension(blocks)
#define __has_blocks 1
#else
#define __has_blocks 0
#endif
    