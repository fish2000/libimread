/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_ENV_HH_
#define LIBIMREAD_INCLUDE_ENV_HH_

#include <mutex>
#include <atomic>
#include <libimread/store.hh>

namespace store {
    
    class env final : public stringmapper {
        
        public:
            virtual bool can_store() const noexcept override;
        
        public:
            env(void);
            env(env const&);
            env(env&&) noexcept;
            virtual ~env();
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            X update(T&& from) {
                store::value_copy(std::forward<T>(from), *this);
            }
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            X update(T&& from, std::string const& prefix,
                               std::string const& sep = ":") {
                store::prefix_copy(std::forward<T>(from), *this, prefix, sep);
            }
        
        public:
            std::string& get_force(std::string const&) const;
        
        public:
            /// implementation of the stringmapper API, in terms of the RocksDB API
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
        protected:
            mutable std::atomic<int> envcount{ 0 };
            static std::mutex mute;
    };
    
} /// namespace store

#endif /// LIBIMREAD_INCLUDE_ENV_HH_