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

/// shortcut getter macro for the RocksDB internal instance
#define SELF() instance.get<rocksdb::DB>()

namespace memory {
    
    /// type-erased deleter callback, using specific RocksDB types
    void killrocks(void*& ptr) {
        rocksdb::DB* local_target = static_cast<rocksdb::DB*>(ptr);
        delete local_target;
        ptr = nullptr;
    }
    
} /// namespace memory

using filesystem::path;
using iterator_ptr_t = std::unique_ptr<rocksdb::Iterator>;

namespace store {
    
    bool rocks::can_store() const noexcept { return true; }
    
    /// RocksDB {Read,Write,Flush}Options static-storage
    struct rocks::options {
        
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
        :rockspth(path::absolute(filepth).str())
        {
            /// background thread-pool customization options from:
            /// https://github.com/facebook/rocksdb/wiki/basic-operations#thread-pools
            auto env = rocksdb::Env::Default();
            env->SetBackgroundThreads(2, rocksdb::Env::LOW);
            env->SetBackgroundThreads(1, rocksdb::Env::HIGH);
            rocksdb::Options options;
            options.env = env;
            options.max_background_compactions = 2;
            options.max_background_flushes = 1;
            /// basic-optimization options from:
            /// https://github.com/facebook/rocksdb/blob/master/examples/simple_example.cc
            options.IncreaseParallelism();
            options.OptimizeLevelStyleCompaction();
            options.create_if_missing = true;
            rocksdb::DB* local_instance = SELF();
            rocksdb::Status status = rocksdb::DB::Open(options, rockspth, &local_instance);
            instance.reset((void*)local_instance);
        }
    
    rocks::~rocks() {
        instance.reset(nullptr);
    }
    
    std::string& rocks::get_force(std::string const& key) const {
        std::string sval;
        rocksdb::Status status = SELF()->Get(options::read(), key, &sval);
        if (status.ok()) {
            cache[key] = sval;
            return cache.at(key);
        }
        return stringmapper::base_t::null_value();
    }
    
    std::string& rocks::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        } else {
            std::string sval;
            rocksdb::Status status = SELF()->Get(options::read(), key, &sval);
            if (status.ok()) {
                cache[key] = sval;
                return cache.at(key);
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    std::string const& rocks::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        } else {
            std::string sval;
            rocksdb::Status status = SELF()->Get(options::read(), key, &sval);
            if (status.ok()) {
                cache[key] = sval;
                return cache.at(key);
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    bool rocks::set(std::string const& key, std::string const& value) {
        rocksdb::Status flushed;
        rocksdb::Status status = SELF()->Put(options::write(), key, value);
        if (status.ok()) {
            flushed = SELF()->Flush(options::flush());
            if (flushed.ok()) {
                if (cache.find(key) != cache.end()) {
                    cache[key] = value;
                } else {
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
        rocksdb::Status status = SELF()->Delete(options::write(), key);
        return status.ok();
    }
    
    std::size_t rocks::count() const {
        uint64_t cnt;
        bool status = SELF()->GetIntProperty(rocksdb::DB::Properties::kEstimateNumKeys, &cnt);
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

#undef SELF