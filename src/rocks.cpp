/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from my own Objective-C++ class:
///     https://gist.github.com/fish2000/b3a7d8accae8d046703f728b4ac82009

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/rocks.hh>
#include <libimread/ext/filesystem/path.h>
#include "rocksdb/env.h"
#include "rocksdb/db.h"

/// Shortcut getter macro for the RocksDB internal instance
#define SELF() instance.get<rocksdb::DB>()

/// Shortcut to std::string{ NULL_STR } value
#define STRINGNULL() stringmapper::base_t::null_value()

namespace memory {
    
    /// type-erased deleter callback, using specific RocksDB types
    void killrocks(void*& ptr) {
        rocksdb::DB* local_target = static_cast<rocksdb::DB*>(ptr);
        delete local_target;
        ptr = nullptr;
    }
    
} /// namespace memory

using rocksdb::Status;
using filesystem::path;
using iterator_ptr_t = std::unique_ptr<rocksdb::Iterator>;

namespace store {
    
    bool rocks::can_store() const noexcept { return true; }
    
    /// RocksDB {Read,Write,Flush}Options static-storage
    struct rocks::options {
        
        static rocksdb::Env* env() {
            /// background thread-pool customization options from:
            /// https://github.com/facebook/rocksdb/wiki/basic-operations#thread-pools
            static rocksdb::Env* env = rocksdb::Env::Default();
            static bool configured = false;
            if (!configured) {
                configured = true;
                env->SetBackgroundThreads(2, rocksdb::Env::LOW);
                env->SetBackgroundThreads(1, rocksdb::Env::HIGH);
            }
            return env;
        }
        
        static rocksdb::Options& opts() {
            /// basic-optimization options from:
            /// https://github.com/facebook/rocksdb/blob/master/examples/simple_example.cc
            static rocksdb::Options o;
            static bool configured = false;
            if (!configured) {
                configured = true;
                o.env = rocks::options::env();
                o.max_background_compactions = 2;
                o.max_background_flushes = 1;
                o.IncreaseParallelism();
                o.OptimizeLevelStyleCompaction();
                o.create_if_missing = true;
            }
            return o;
        }
        
        static rocksdb::ReadOptions& read() {
            static rocksdb::ReadOptions ro;
            return ro;
        }
        
        static rocksdb::WriteOptions& write() {
            static rocksdb::WriteOptions wo;
            return wo;
        }
        
        static rocksdb::FlushOptions& flush() {
            static rocksdb::FlushOptions fo;
            return fo;
        }
        
    };
    
    /// Stortcuts to flags defined in rocksdb::DB::Properties
    struct rocks::flags {
        
        #define FLAG(flagname)                                                      \
            __attribute__((__warn_unused_result__))                                 \
            static decltype(rocksdb::DB::Properties::flagname)& flagname() {        \
                return      rocksdb::DB::Properties::flagname;                      \
            }
        
        FLAG(kSizeAllMemTables);
        FLAG(kEstimateNumKeys);
        
    };
    
    /// copy constructor
    rocks::rocks(rocks const& other)
        :instance(other.instance)
        ,rockspth(other.rockspth)
        {}
    
    /// move constructor
    rocks::rocks(rocks&& other) noexcept
        :instance(std::move(other.instance))
        ,rockspth(std::move(other.rockspth))
        {}
    
    /// database file path constructor
    rocks::rocks(std::string const& filepth)
        :rockspth(path::expand_user(filepth).make_absolute().str())
        {
            rocksdb::DB* local_instance = SELF();
            Status status = rocksdb::DB::Open(options::opts(),
                                              rockspth,
                                              &local_instance);
            instance.reset((void*)local_instance);
        }
    
    rocks::~rocks() {}
    
    std::string& rocks::get_force(std::string const& key) const {
        std::string sval;
        Status status = SELF()->Get(options::read(), key, &sval);
        if (status.ok()) {
            cache[key] = sval;
            return cache.at(key);
        }
        return STRINGNULL();
    }
    
    std::string const& rocks::filepath() const {
        return rockspth;
    }
    
    std::size_t rocks::memorysize() const {
        uint64_t memsize = 0;
        bool status = SELF()->GetAggregatedIntProperty(flags::kSizeAllMemTables(), &memsize);
        if (status) { return static_cast<std::size_t>(memsize); }
        return 0;
    }
    
    std::string& rocks::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        }
        return get_force(key);
    }
    
    std::string const& rocks::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        }
        return get_force(key);
    }
    
    bool rocks::set(std::string const& key, std::string const& value) {
        Status flushed;
        Status status = value != STRINGNULL() ? SELF()->Put(options::write(), key, value)
                                              : SELF()->Delete(options::write(), key);
        if (status.ok()) {
            flushed = SELF()->Flush(options::flush());
            if (flushed.ok()) {
                if (cache.find(key) != cache.end()) {
                    if (value == STRINGNULL()) {
                        cache.erase(key);
                    } else {
                        cache[key] = value;
                    }
                } else if (value != STRINGNULL()) {
                    cache.insert({ key, value });
                }
            }
        }
        return status.ok() && flushed.ok();
    }
    
    bool rocks::del(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            cache.erase(key);
        }
        Status status = SELF()->Delete(options::write(), key);
        return status.ok();
    }
    
    std::size_t rocks::count() const {
        uint64_t cnt;
        bool status = SELF()->GetIntProperty(flags::kEstimateNumKeys(), &cnt);
        if (status) { return static_cast<std::size_t>(cnt); }
        return 0;
    }
    
    stringmapper::stringvec_t rocks::list() const {
        iterator_ptr_t it(SELF()->NewIterator(options::read()));
        stringmapper::stringvec_t out{};
        out.reserve(count());
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            out.emplace_back(it->key().ToString());
        }
        return out;
    }
    
} /// namespace store

#undef STRINGNULL
#undef SELF
#undef FLAG