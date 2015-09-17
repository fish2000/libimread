
#include <algorithm>
#include <libimread/ext/memory/refcount.hh>

namespace std {
    
    template <>
    void swap(Guid& guid0, Guid& guid1) {
        guid0.swap(guid1);
    }
    
    template <typename T>
    void swap(RefCount<T>& refcount0, RefCount<T>& refcount1) {
        refcount0.swap(refcount1);
    }
    
}; /* namespace std */

namespace memory {
    
    void garbageday() {
        /// ""?ref-0p-="" -Tux
        using KVPair = typename decltype(refcounts)::value_type;
        std::for_each(refcounts.begin(),
                      refcounts.end(), [](const KVPair& kvpair) {
            if (kvpair.second.load() < 1) {
                refcounts.erase(kvpair.first);
            }
        });
    }
    
}; /* namespace memory */

