
#include <deque>
#include <algorithm>
#include <libimread/ext/memory/refcount.hh>

namespace memory {
    
    void garbageday() {
        /// ""?ref-0p-="" -Tux
        using Key = typename decltype(refcounts)::key_type;
        using KVPair = typename decltype(refcounts)::value_type;
        std::deque<Key> keyjail;
        
        std::for_each(refcounts.begin(),
                      refcounts.end(), [&keyjail](const KVPair& kvpair) {
            if (kvpair.second.load() < 1) {
                keyjail.push_back(kvpair.first);
            }
        });
        
        /// if we were to lock the entire master-refcount list before deleting,
        /// this loop'd be around which part we'd position whatever such locking/semaphore
        /// guard-ish kind of struct stuff... that we'd be employing. Like for all the shit here.
        for (auto&& key : keyjail) {
            refcounts.erase(key);
        }
        keyjail.clear();
    }
    
}; /* namespace memory */

