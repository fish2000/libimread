
#ifndef IMREAD_OPTIONS_HH
#define IMREAD_OPTIONS_HH

#include <string>
#include <cstring>
#include <tuple>
#include <unordered_map>

#include <type_traits>
#include <iod/sio.hh>
#include <iod/callable_traits.hh>
#include <iod/tuple_utils.hh>
#include <iod/utils.hh>
#include <iod/bind_method.hh>
#include <libimread/symbols.hh>

namespace im {
    
    /*
    using Symbolizer = iod::D_caller;
    template <typename ...O>
    struct Option : public Symbolizer {
        auto out;
        
        Option() = delete;
        Option(O... opts)
            :Symbolizer()
            { out = operator()(opts...); }
        Option &operator=(const auto &whatevs) { return out; }
    };
    */
    
    using iod::D;
    
    // template <typename ...O>
    // auto options(O&&... opts) { return D(opts...); }
    // #define Options(...) auto options = D(__VA_ARGS__)
    
    /// number_or_string is a sort of typed union.
    /// We could have used boost::any here, but that would have brought in a big
    /// dependency, which would otherwise not be used.
    struct number_or_string {
        number_or_string()
            :holds_(ns_empty)
            { }
    
        explicit number_or_string(std::string s)
            :str_(s), holds_(ns_string)
            { }
        explicit number_or_string(int i)
            :int_(i), holds_(ns_int)
            { }
        explicit number_or_string(double v)
            :double_(v), holds_(ns_double)
            { }
        
        bool get_int(int &n) const {
            if (holds_ != ns_int) { return false; }
            n = int_;
            return true; 
        }
        bool get_double(double &n) const {
            if (holds_ != ns_double) { return false; }
            n = double_;
            return true;
        }
        bool get_str(std::string &s) const {
            if (holds_ != ns_string) { return false; }
            s = str_;
            return true;
        }
        const char *maybe_c_str() const {
            if (holds_ == ns_string) { return str_.c_str(); }
            return 0;
        }
    
        private:
            std::string str_;
            int int_;
            double double_;
            enum { ns_empty, ns_string, ns_int, ns_double } holds_;
    };

    typedef std::unordered_map<std::string, number_or_string> options_map;

    inline const char *get_optional_cstring(const options_map &opts, const std::string key) {
        options_map::const_iterator iter = opts.find(key);
        if (iter == opts.end()) { return 0; }
        return iter->second.maybe_c_str();
    }

    inline int get_optional_int(const options_map &opts, const std::string key, const int def) {
        options_map::const_iterator iter = opts.find(key);
        if (iter == opts.end()) { return def; }
        int v;
        if (iter->second.get_int(v)) { return v; }
        return def;
    }

    inline bool get_optional_bool(const options_map &opts, const std::string key, const bool def) {
        return get_optional_int(opts, key, def);
    }

}

#endif /// IMREAD_OPTIONS_HH