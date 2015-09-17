
#include <algorithm>
#include <libimread/ext/memory/refcount.hh>

namespace std {
    
    template <>
    void swap(Guid& guid0, Guid& guid1) {
        guid0.swap(guid1);
    }
    
}; /* namespace std */

namespace memory {
    
    // std::unordered_map<Guid, std::atomic<int64_t>> refcounts;
    
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

