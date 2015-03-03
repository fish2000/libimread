/*
 *  spx_defines.h
 *
 *  Created by Jean-Daniel Dupas.
 *  Copyright © 2013 Jean-Daniel Dupas. All rights reserved.
 *
 *  File version: 115
 *  File Generated using “basegen --name=SharedPrefix --prefix=spx --objc --cxx”.
 */

#if !defined(SPX_DEFINES_H__)
#define SPX_DEFINES_H__ 1

// MARK: Clang Macros
#ifndef __has_builtin
  #define __has_builtin(x) __has_builtin_ ## x
#endif

#ifndef __has_attribute
  #define __has_attribute(x) __has_attribute_ ## x
#endif

#ifndef __has_feature
  #define __has_feature(x) __has_feature_ ## x
#endif

#ifndef __has_extension
  #define __has_extension(x) __has_feature(x)
#endif

#ifndef __has_include
  #define __has_include(x) 0
#endif

#ifndef __has_include_next
  #define __has_include_next(x) 0
#endif

#ifndef __has_warning
  #define __has_warning(x) 0
#endif

// MARK: Visibility
#if defined(_WIN32)
  #define SPX_HIDDEN

  #if defined(SPX_STATIC_LIBRARY)
      #define SPX_VISIBLE
  #else
    #if defined(SPX_DLL_EXPORT)
      #define SPX_VISIBLE __declspec(dllexport)
    #else
      #define SPX_VISIBLE __declspec(dllimport)
    #endif
  #endif
#endif

#if !defined(SPX_VISIBLE)
  #define SPX_VISIBLE __attribute__((__visibility__("default")))
#endif

#if !defined(SPX_HIDDEN)
  #define SPX_HIDDEN __attribute__((__visibility__("hidden")))
#endif

#if !defined(SPX_EXTERN)
  #if defined(__cplusplus)
    #define SPX_EXTERN extern "C"
  #else
    #define SPX_EXTERN extern
  #endif
#endif

/* SPX_EXPORT SPX_PRIVATE should be used on
 extern variables and functions declarations */
#if !defined(SPX_EXPORT)
  #define SPX_EXPORT SPX_EXTERN SPX_VISIBLE
#endif

#if !defined(SPX_PRIVATE)
  #define SPX_PRIVATE SPX_EXTERN SPX_HIDDEN
#endif

// MARK: Inline
#if defined(__cplusplus) && !defined(__inline__)
  #define __inline__ inline
#endif

#if !defined(SPX_INLINE)
  #if !defined(__NO_INLINE__)
    #if defined(_MSC_VER)
      #define SPX_INLINE __forceinline static
    #else
      #define SPX_INLINE __inline__ __attribute__((__always_inline__)) static
    #endif
  #else
    #define SPX_INLINE __inline__ static
  #endif /* No inline */
#endif

// MARK: Attributes
#if !defined(SPX_NORETURN)
  #if defined(_MSC_VER)
    #define SPX_NORETURN __declspec(noreturn)
  #else
    #define SPX_NORETURN __attribute__((__noreturn__))
  #endif
#endif

#if !defined(SPX_DEPRECATED)
  #if defined(_MSC_VER)
    #define SPX_DEPRECATED(msg) __declspec(deprecated(msg))
  #elif defined(__clang__)
    #define SPX_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
  #else
    #define SPX_DEPRECATED(msg) __attribute__((__deprecated__))
  #endif
#endif

#if !defined(SPX_UNUSED)
  #if defined(_MSC_VER)
    #define SPX_UNUSED
  #else
    #define SPX_UNUSED __attribute__((__unused__))
  #endif
#endif

#if !defined(SPX_REQUIRES_NIL_TERMINATION)
  #if defined(_MSC_VER)
    #define SPX_REQUIRES_NIL_TERMINATION
  #elif defined(__APPLE_CC__) && (__APPLE_CC__ >= 5549)
    #define SPX_REQUIRES_NIL_TERMINATION __attribute__((__sentinel__(0,1)))
  #else
    #define SPX_REQUIRES_NIL_TERMINATION __attribute__((__sentinel__))
  #endif
#endif

#if !defined(SPX_REQUIRED_ARGS)
  #if defined(_MSC_VER)
    #define SPX_REQUIRED_ARGS(idx, ...)
  #else
    #define SPX_REQUIRED_ARGS(idx, ...) __attribute__((__nonnull__(idx, ##__VA_ARGS__)))
  #endif
#endif

#if !defined(SPX_FORMAT)
  #if defined(_MSC_VER)
    #define SPX_FORMAT(fmtarg, firstvararg)
  #else
    #define SPX_FORMAT(fmtarg, firstvararg) __attribute__((__format__ (__printf__, fmtarg, firstvararg)))
  #endif
#endif

#if !defined(SPX_CF_FORMAT)
  #if defined(__clang__)
    #define SPX_CF_FORMAT(i, j) __attribute__((__format__(__CFString__, i, j)))
  #else
    #define SPX_CF_FORMAT(i, j)
  #endif
#endif

#if !defined(SPX_NS_FORMAT)
  #if defined(__clang__)
    #define SPX_NS_FORMAT(i, j) __attribute__((__format__(__NSString__, i, j)))
  #else
    #define SPX_NS_FORMAT(i, j)
  #endif
#endif

// MARK: -
// MARK: Static Analyzer
#ifndef CF_CONSUMED
  #if __has_attribute(cf_consumed)
    #define CF_CONSUMED __attribute__((__cf_consumed__))
  #else
    #define CF_CONSUMED
  #endif
#endif

#ifndef CF_RETURNS_RETAINED
  #if __has_attribute(cf_returns_retained)
    #define CF_RETURNS_RETAINED __attribute__((__cf_returns_retained__))
  #else
    #define CF_RETURNS_RETAINED
  #endif
#endif

#ifndef CF_RETURNS_NOT_RETAINED
	#if __has_attribute(cf_returns_not_retained)
		#define CF_RETURNS_NOT_RETAINED __attribute__((__cf_returns_not_retained__))
	#else
		#define CF_RETURNS_NOT_RETAINED
	#endif
#endif

#ifndef CF_IMPLICIT_BRIDGING_ENABLED
  #if __has_feature(arc_cf_code_audited)
    #define CF_IMPLICIT_BRIDGING_ENABLED _Pragma("clang arc_cf_code_audited begin")
  #else
    #define CF_IMPLICIT_BRIDGING_ENABLED
  #endif
#endif

#ifndef CF_IMPLICIT_BRIDGING_DISABLED
  #if __has_feature(arc_cf_code_audited)
    #define CF_IMPLICIT_BRIDGING_DISABLED _Pragma("clang arc_cf_code_audited end")
  #else
    #define CF_IMPLICIT_BRIDGING_DISABLED
  #endif
#endif


#if defined(__cplusplus)

/* SPX_CXX_EXPORT and SPX_CXX_PRIVATE can be used
 to define C++ classes visibility. */
#if defined(__cplusplus)
  #if !defined(SPX_CXX_PRIVATE)
    #define SPX_CXX_PRIVATE SPX_HIDDEN
  #endif

  #if !defined(SPX_CXX_EXPORT)
    #define SPX_CXX_EXPORT SPX_VISIBLE
  #endif
#endif

// GCC 4.x C++11 support
#if defined(__GNUC__) && !defined(__clang__) && !defined(__gcc_features_defined)
#define __gcc_features_defined 1

#if defined(__GXX_RTTI)
#  define __has_feature_cxx_rtti 1
#endif

#if defined(__EXCEPTIONS)
#  define __has_feature_cxx_exceptions 1
#endif

#define SPX_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

// GCC 4.3
#if SPX_GCC_VERSION >= 40300
#  define __has_feature_cxx_static_assert 1
#  define __has_feature_cxx_rvalue_references 1
#endif

// GCC 4.4
#if SPX_GCC_VERSION >= 40400
#  define __has_feature_cxx_auto_type 1
#  define __has_feature_cxx_deleted_functions 1
#  define __has_feature_cxx_defaulted_functions 1
#endif

// GCC 4.5
#if SPX_GCC_VERSION >= 40500
#  define __has_feature_cxx_alignof 1
#  define __has_feature_cxx_lambdas 1
#  define __has_feature_cxx_decltype 1
#  define __has_feature_cxx_explicit_conversions 1
#endif

// GCC 4.6
#if SPX_GCC_VERSION >= 40600
#  define __has_feature_cxx_nullptr 1
#  define __has_feature_cxx_noexcept 1
#  define __has_feature_cxx_constexpr 1
#  define __has_feature_cxx_range_for 1
#endif

// GCC 4.7
#if SPX_GCC_VERSION >= 40700
#  define __has_feature_cxx_override_control 1
#  define __has_feature_cxx_delegating_constructors 1
#endif

// GCC 4.8
#if SPX_GCC_VERSION >= 40800
#  define __has_feature_cxx_alignas 1
#  define __has_feature_cxx_inheriting_constructors 1
#endif

#undef SPX_GCC_VERSION

#endif

#if defined(_MSC_VER) && !defined(__msc_features_defined)
#define __msc_features_defined 1

#define __has_builtin___debugbreak 1

// VisualStudio 2010
#if _MSC_VER >= 1600
  #define __has_feature_cxx_nullptr 1
  #define __has_feature_cxx_auto_type 1
  #define __has_feature_cxx_static_assert 1
  #define __has_feature_cxx_trailing_return 1
  #define __has_feature_cxx_override_control 1
  #define __has_feature_cxx_rvalue_references 1
  #define __has_feature_cxx_local_type_template_args 1
#endif

// VisualStudio 2011
#if _MSC_VER >= 1700
  #define __has_feature_cxx_lambdas 1
  #define __has_feature_cxx_decltype 1
  #define __has_feature_cxx_range_for 1
#endif

#endif /* _MSC_VER */

// MARK: C++ 2011
#if __has_extension(cxx_override_control)
  #if !defined(_MSC_VER) || _MSC_VER >= 1700
    #define spx_final final
  #else
    #define spx_final sealed
  #endif
  #define spx_override override
#else
  // not supported
  #define spx_final
  #define spx_override
#endif

#if __has_extension(cxx_nullptr)
  #undef NULL
  #define NULL nullptr
#else
  // use the standard declaration
#endif

#if __has_extension(cxx_noexcept)
  #define spx_noexcept noexcept
  #define spx_noexcept_(arg) noexcept(arg)
#else
  #define spx_noexcept
  #define spx_noexcept_(arg)
#endif

#if __has_extension(cxx_constexpr)
  #define spx_constexpr constexpr
#else
  #define spx_constexpr
#endif

#if __has_extension(cxx_rvalue_references)
  /* declaration for move, swap, forward, ... */
  #define spx_move(arg) std::move(arg)
  #define spx_forward(Ty, arg) std::forward<Ty>(arg)
#else
  #define spx_move(arg) arg
  #define spx_forward(Ty, arg) arg
#endif

#if __has_extension(cxx_deleted_functions)
  #define spx_deleted = delete
#else
  #define spx_deleted
#endif

#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
  #define spx_fallthrough [[clang::fallthrough]]
#else
  #define spx_fallthrough do {} while (0)
#endif

// MARK: Other C++ macros

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#ifndef SPX_DISALLOW_COPY_AND_ASSIGN
  #define SPX_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    private: \
     TypeName(const TypeName&) spx_deleted; \
     void operator=(const TypeName&) spx_deleted
#endif

#ifndef SPX_DISALLOW_MOVE
  #if __has_extension(cxx_rvalue_references)
    #define SPX_DISALLOW_MOVE(TypeName) \
      private: \
       TypeName(TypeName&&) spx_deleted; \
       void operator=(TypeName&&) spx_deleted
  #else
    #define SPX_DISALLOW_MOVE(TypeName)
  #endif
#endif

#ifndef SPX_DISALLOW_COPY_ASSIGN_MOVE
  #define SPX_DISALLOW_COPY_ASSIGN_MOVE(TypeName) \
    SPX_DISALLOW_MOVE(TypeName);                  \
    SPX_DISALLOW_COPY_AND_ASSIGN(TypeName)
#endif

#endif /* __cplusplus */

#if defined(__OBJC__)

/* SPX_OBJC_EXPORT and SPX_OBJC_PRIVATE can be used
 to define ObjC classes visibility. */
#if !defined(SPX_OBJC_PRIVATE)
  #if __LP64__
    #define SPX_OBJC_PRIVATE SPX_HIDDEN
  #else
    #define SPX_OBJC_PRIVATE
  #endif /* 64 bits runtime */
#endif

#if !defined(SPX_OBJC_EXPORT)
  #if __LP64__
    #define SPX_OBJC_EXPORT SPX_VISIBLE
  #else
    #define SPX_OBJC_EXPORT
  #endif /* 64 bits runtime */
#endif

// MARK: Static Analyzer
#ifndef SPX_UNUSED_IVAR
  #if __has_extension(attribute_objc_ivar_unused)
    #define SPX_UNUSED_IVAR __attribute__((__unused__))
  #else
    #define SPX_UNUSED_IVAR
  #endif
#endif

#ifndef NS_CONSUMED
  #if __has_attribute(ns_consumed)
    #define NS_CONSUMED __attribute__((__ns_consumed__))
  #else
    #define NS_CONSUMED
  #endif
#endif

#ifndef NS_CONSUMES_SELF
  #if __has_attribute(ns_consumes_self)
    #define NS_CONSUMES_SELF __attribute__((__ns_consumes_self__))
  #else
    #define NS_CONSUMES_SELF
  #endif
#endif

#ifndef NS_RETURNS_RETAINED
  #if __has_attribute(ns_returns_retained)
    #define NS_RETURNS_RETAINED __attribute__((__ns_returns_retained__))
  #else
    #define NS_RETURNS_RETAINED
  #endif
#endif

#ifndef NS_RETURNS_NOT_RETAINED
  #if __has_attribute(ns_returns_not_retained)
    #define NS_RETURNS_NOT_RETAINED __attribute__((__ns_returns_not_retained__))
  #else
    #define NS_RETURNS_NOT_RETAINED
  #endif
#endif

#ifndef NS_RETURNS_AUTORELEASED
  #if __has_attribute(ns_returns_autoreleased)
    #define NS_RETURNS_AUTORELEASED __attribute__((__ns_returns_autoreleased__))
  #else
    #define NS_RETURNS_AUTORELEASED
  #endif
#endif

/* Method Family */
#ifndef NS_METHOD_FAMILY
  /* supported families are: none, alloc, copy, init, mutableCopy, and new. */
  #if __has_attribute(ns_returns_autoreleased)
    #define NS_METHOD_FAMILY(family) __attribute__((objc_method_family(family)))
  #else
    #define NS_METHOD_FAMILY(arg)
  #endif
#endif

// gracefully degrade
#if !__has_feature(objc_instancetype)
  #define instancetype id
#endif

#endif /* ObjC */


#endif /* SPX_DEFINES_H__ */
