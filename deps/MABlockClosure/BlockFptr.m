
#import "BlockFptr.h"

void *BlockFptr(id block)
{
    @synchronized(block)
    {
        MABlockClosure *closure = objc_getAssociatedObject(block, BlockFptr);
        if(!closure)
        {
            closure = [[MABlockClosure alloc] initWithBlock: block];
            objc_setAssociatedObject(block, BlockFptr, closure, OBJC_ASSOCIATION_RETAIN);
            [closure release]; // retained by the associated object assignment
        }
        return [closure fptr];
    }
}

void *BlockFptrAuto(id block)
{
    return BlockFptr([[block copy] autorelease]);
}
