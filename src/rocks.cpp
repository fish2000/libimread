/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from my own Objective-C++ class:
///     https://gist.github.com/fish2000/b3a7d8accae8d046703f728b4ac82009

#include <libimread/libimread.hpp>
#include <libimread/rocks.hh>
#include <libimread/ext/filesystem/path.h>
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

namespace store {
    
    bool rocks::can_store() const noexcept { return true; }
    
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
            rocksdb::Options options;
            options.create_if_missing = true;
            rocksdb::DB* local_instance = SELF();
            rocksdb::Status status = rocksdb::DB::Open(options, rockspth, &local_instance);
            instance.reset((void*)local_instance);
        }
    
    rocks::~rocks() {
        rocksdb::DB* local_instance = SELF();
        delete local_instance;
        instance.reset(nullptr);
    }
    
    std::string& rocks::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            std::string sval;
            rocksdb::Status status = SELF()->Get(rocksdb::ReadOptions(), key, &sval);
            if (status.ok()) {
                cache[key] = sval;
                return cache[key];
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    std::string const& rocks::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            std::string sval;
            rocksdb::Status status = SELF()->Get(rocksdb::ReadOptions(), key, &sval);
            if (status.ok()) {
                cache[key] = sval;
                return cache[key];
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    bool rocks::set(std::string const& key, std::string const& value) {
        rocksdb::Status status = SELF()->Put(rocksdb::WriteOptions(), key, value);
        if (status.ok()) {
            cache[key] = value;
            rocksdb::Status flushed = SELF()->Flush(rocksdb::FlushOptions());
        }
        return status.ok();
    }
    
    bool rocks::del(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            cache.erase(key);
        }
        rocksdb::Status status = SELF()->Delete(rocksdb::WriteOptions(), key);
        return status.ok();
    }
    
    std::size_t rocks::count() const {
        uint64_t cnt;
        bool status = SELF()->GetIntProperty(rocksdb::DB::Properties::kEstimateNumKeys, &cnt);
        if (status) { return static_cast<std::size_t>(cnt); }
        return 0;
    }
    
    stringmapper::stringvec_t rocks::list() const {
        rocksdb::Iterator* it = SELF()->NewIterator(rocksdb::ReadOptions());
        stringmapper::stringvec_t out{};
        out.reserve(count());
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            out.emplace_back(it->value().ToString());
        }
        delete it;
        return out;
    }
    
} /// namespace store

#undef SELF