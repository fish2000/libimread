
#include <deque>
#include <algorithm>
#include <libimread/ext/memory/refcount.hh>

namespace memory {
    
    GuidGenerator generator = GuidGenerator();
    refcount_t refcounts{};
    // thread_t garbagecollector(garbageday);
    std::mutex mute;
    
    using Key = typename decltype(refcounts)::key_type;
    using KVPair = typename decltype(refcounts)::value_type;
    
    void garbageday() {
        /// ""?ref-0p-="" -Tux
        std::lock_guard<std::mutex> lock(mute);
        std::deque<Key> keyjail;
        std::for_each(refcounts.begin(),
                      refcounts.end(), [&keyjail](KVPair const& kvpair) {
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

