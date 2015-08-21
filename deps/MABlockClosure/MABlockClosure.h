
#import <Foundation/Foundation.h>

#if (TARGET_OS_IPHONE && TARGET_OS_EMBEDDED) || TARGET_IPHONE_SIMULATOR
#define USE_CUSTOM_LIBFFI 1
#endif

#if USE_CUSTOM_LIBFFI
#import <ffi.h>
#define USE_LIBFFI_CLOSURE_ALLOC 1
#else // use system libffi
#import <ffi/ffi.h>
#endif


@interface MABlockClosure : NSObject
{
    NSMutableArray *_allocations;
    ffi_cif _closureCIF;
    ffi_cif _innerCIF;
    int _closureArgCount;
    ffi_closure *_closure;
    void *_closureFptr;
    id _block;
}

- (id)initWithBlock: (id)block;

- (void *)fptr;

@end
