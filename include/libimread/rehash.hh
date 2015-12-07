/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_REHASH_HH_
#define LIBIMREAD_REHASH_HH_

#include <cstdlib>
#include <type_traits>
#include <functional>

/// hashing trick cribbed from boost,
/// via http://stackoverflow.com/a/23860042/298171 --
/// the REHASHER() macro provides the actual hash-in implement.

#ifndef REHASHER
#define REHASHER(seed, hasher, value) \
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2)
#endif

namespace hash {
    
    /// Calculates the hash value of a thingy and "hashes in"
    /// this value to the seed value, which it modifies in-place
    
    template <typename T> inline
    void rehash(std::size_t& seed, const T& v) {
        std::hash<T> hasher;
        REHASHER(seed, hasher, v);
    }
    
    /// BinaryOperation-style functor to "hash in" a thingy's hash value
    /// to the provided seed, which it returns -- this can be used, say,
    /// with std::accumulate() to reduce an interable of hashable thingies
    /// to a single unique hash (q.v. filesystem::path::hash() sub.)
    
    template <typename T,
              typename SeedT = std::size_t>
    struct rehasher {
        using seed_t = SeedT;
        using hasher_t = std::hash<std::remove_cv_t<T>>;
        using hashee_t = std::add_lvalue_reference_t<std::add_const_t<T>>;
        hasher_t hasher; /// default construction
        
        seed_t operator()(seed_t seed, hashee_t v) {
            REHASHER(seed, hasher, v);
            return seed;
        }
    };
    
}

#undef REHASHER

#endif /// LIBIMREAD_REHASH_HH_