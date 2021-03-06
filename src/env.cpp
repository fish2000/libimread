/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from my own Objective-C++ class:
///     https://gist.github.com/fish2000/b3a7d8accae8d046703f728b4ac82009

#include <cstdlib>
#include <libimread/libimread.hpp>
#include <libimread/env.hh>
#include <libimread/ext/pystring.hh>

/// Shortcut to std::string{ NULL_STR } value
#define STRINGNULL() stringmapper::base_t::null_value()

/// the environment array -- q.v. note sub:
/// http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap08.html#tag_08
extern char** environ;

namespace store {
    
    bool env::can_store() const noexcept { return true; }
    
    /// whatever-dogg constructor
    env::env(void) {
        count();
    }
    
    /// copy constructor
    env::env(env const& other) {
        envcount.store(other.envcount.load());
        cache = other.cache;
    }
    
    /// move constructor
    env::env(env&& other) noexcept {
        envcount.store(other.envcount.load());
        cache = std::move(other.cache);
    }
    
    env::~env() {}
    
    std::string& env::get_force(std::string const& key) const {
        char* cvalue = nullptr;
        {
            std::unique_lock<std::mutex> getlock(getmute);
            cvalue = std::getenv(key.c_str());
        }
        if (cvalue != nullptr) {
            std::string value(cvalue);
            if (value.size() > 0) {
                {
                    std::unique_lock<std::mutex> setlock(setmute);
                    cache[key] = value;
                }
                return cache.at(key);
            }
        }
        return STRINGNULL();
    }
    
    std::string& env::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        }
        return get_force(key);
    }
    
    std::string const& env::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache.at(key);
        }
        return get_force(key);
    }
    
    bool env::set(std::string const& key, std::string const& value) {
        int status;
        {
            std::lock(setmute, getmute);
            std::lock_guard<std::mutex> setlock(setmute, std::adopt_lock);
            std::lock_guard<std::mutex> getlock(getmute, std::adopt_lock);
            status = ::setenv(key.c_str(), value.c_str(), 1);
        }
        if (status == 0) {
            if (cache.find(key) != cache.end()) {
                std::unique_lock<std::mutex> setlock(setmute);
                cache[key] = value;
            } else {
                std::unique_lock<std::mutex> setlock(setmute);
                cache.insert({ key, value });
                ++envcount;
            }
            return true;
        }
        return false;
    }
    
    bool env::del(std::string const& key) {
        bool unset = false;
        {
            std::lock(setmute, getmute);
            std::lock_guard<std::mutex> setlock(setmute, std::adopt_lock);
            std::lock_guard<std::mutex> getlock(getmute, std::adopt_lock);
            unset = (::unsetenv(key.c_str()) == 0);
        }
        if (cache.find(key) != cache.end()) {
            std::unique_lock<std::mutex> setlock(setmute);
            cache.erase(key);
        }
        if (unset) { --envcount; }
        return unset;
    }
    
    std::size_t env::count() const {
        std::size_t idx = 0;
        {
            std::unique_lock<std::mutex> getlock(getmute);
            while (environ[idx]) { ++idx; }
        }
        envcount.store(idx);
        return idx;
    }
    
    stringmapper::stringvec_t env::list() const {
        stringmapper::stringvec_t out{};
        stringmapper::stringvec_t parts{};
        std::size_t idx = 0;
        out.reserve(count());
        {
            std::unique_lock<std::mutex> getlock(getmute);
            while (environ[idx]) {
                std::string envkv(environ[idx++]);
                pystring::split(envkv, parts, "=");
                out.emplace_back(parts.front());
            }
            envcount.store(idx);
        }
        return out;
    }
    
    /// define static mutex declared within store::env :
    std::mutex env::getmute;
    std::mutex env::setmute;
    
} /// namespace store