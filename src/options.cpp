/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/options.hh>

#define STRINGNULL() stringmapper::base_t::null_value()

namespace im {
    
    #pragma mark -
    #pragma mark method implementations for im::OptionsList
    
    OptionsList::OptionsList()
        :Json()
        { mkarray(); }
    
    OptionsList::OptionsList(stringvec_t const& stringvec) {
        Json::Array* array_ptr = mkarray();
        for (std::string const& string : stringvec) {
            Json jstring(string);
            array_ptr->add(jstring.root);
        }
    }
    
    OptionsList::OptionsList(stringvec_t&& stringvec) noexcept {
        Json::Array* array_ptr = mkarray();
        for (std::string const& string : stringvec) {
            Json jstring(string);
            array_ptr->add(jstring.root);
        }
    }
    
    OptionsList::OptionsList(Json const& other)
        :Json(other)
        { mkarray(); }
    
    OptionsList::OptionsList(Json&& other) noexcept
        :Json(std::move(other))
        { mkarray(); }
    
    /// convert from `im::Options const&`: use only the key list
    OptionsList::OptionsList(Options const& other)
        :OptionsList(other.list())
        {}
    
    /// convert from `im::Options&&`: use only the key list
    OptionsList::OptionsList(Options&& other) noexcept
        :OptionsList(other.list())
        {}
    
    OptionsList::OptionsList(OptionsList const& other)
        :OptionsList()
        {
            root->unref();
            Node* node = other.root;
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
    
    OptionsList::OptionsList(OptionsList&& other) noexcept
        :OptionsList()
        {
            Node* node = std::exchange(other.root, root);
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
    
    OptionsList::OptionsList(std::istream& is, bool full)
        :Json(is, full)
        { mkarray(); }
    
    OptionsList::OptionsList(string_init_t string_init)
        :OptionsList(stringvec_t(string_init.begin(),
                                 string_init.end()))
        {}
    
    /// using the initializer list type meant for `im::Options`:
    OptionsList::OptionsList(stringpair_init_t stringpair_init)
        :OptionsList(Options(stringpair_init))
        {}
    
    OptionsList::~OptionsList() {}
    
    /// member swap
    void OptionsList::swap(OptionsList& other) noexcept {
        using std::swap;
        swap(root, other.root);
    }
    
    /// friend swap
    void swap(OptionsList& lhs, OptionsList& rhs) noexcept {
        lhs.swap(rhs);
    }
    
    OptionsList& OptionsList::operator=(string_init_t string_init) {
        OptionsList(string_init).swap(*this);
        return *this;
    }
    
    /// using the initializer list type meant for `im::Options`:
    OptionsList& OptionsList::operator=(stringpair_init_t stringpair_init) {
        OptionsList(stringpair_init).swap(*this);
        return *this;
    }
    
    OptionsList& OptionsList::operator=(Json const& json) {
        OptionsList(json).swap(*this);
        return *this;
    }
    
    OptionsList& OptionsList::operator=(Json&& json) noexcept {
        Node* node = std::exchange(json.root, root);
        (root = (node == nullptr ? &Node::null : node))->refcnt++;
        return *this;
    }
    
    OptionsList& OptionsList::operator=(OptionsList const& other) {
        OptionsList(other).swap(*this);
        return *this;
    }
    
    OptionsList& OptionsList::operator=(OptionsList&& other) noexcept {
        Node* node = std::exchange(other.root, root);
        (root = (node == nullptr ? &Node::null : node))->refcnt++;
        return *this;
    }
    
    bool OptionsList::can_store() const noexcept { return false; }
    
    OptionsList OptionsList::parse(std::string const& str) {
        std::istringstream is(str);
        OptionsList parsed(is);
        if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
        while (std::isspace(is.get()))
            /* skip */;
        if (is.eof()) { return parsed; }
        throw Json::parse_error("JSON im::OptionsList format error", is);
    }
    
    std::size_t OptionsList::count() const {
        return Json::size();
    }
    
    #pragma mark -
    #pragma mark method implementations for im::Options
    
    /// default constructor -- calls Json::mkobject, which
    /// initializes the root node as a Json::Object:
    Options::Options()
        :Json()
        {
            mkobject();
        }
    
    /// protected delegate stringmap_t copy constructor:
    Options::Options(stringmap_t const& stringmap) {
        mkobject();
        cache = stringmap_t(stringmap.begin(),
                            stringmap.end());
    }
    
    /// protected delegate stringmap_t move constructor:
    Options::Options(stringmap_t&& stringmap) noexcept {
        mkobject();
        cache = std::move(stringmap);
    }
    
    /// Copy-construct from a JSON value:
    Options::Options(Json const& json)
        :Json(json)
        {
            mkobject();
        }
    
    /// Move-construct from a JSON value:
    Options::Options(Json&& json) noexcept
        :Json(std::move(json))
        {
            mkobject();
        }
    
    /// Copy-constructor:
    Options::Options(Options const& other)
        :Options(other.cache)
        {
            root->unref();
            Node* node = other.root;
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
    
    /// Move constructor:
    Options::Options(Options&& other) noexcept
        :Options(std::move(other.cache))
        {
            Node* node = std::exchange(other.root, root);
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
    
    /// Input-stream constructor, for JSON stream-parsing:
    Options::Options(std::istream& is, bool full)
        :Json(is, full)
        {}
    
    /// initializer-list constructor -- allows “literals” e.g.
    /// 
    /// Options opts = {
    ///     {   "yo", "dogg" },
    ///     {    "i", "heard" },
    ///     {  "you", "like" },
    ///     { "list", "initialization" }
    /// };
    ///
    /// … the argument type is actually std::initializer_list<
    ///                                             std::pair<std::string,
    ///                                                       std::string>>
    /// … N.B. we need to add a version that accepts Json values,
    ///        in place of the pairs’ second std::string
    Options::Options(stringpair_init_t stringpair_init)
        :Options(stringmap_t(stringpair_init.begin(),
                             stringpair_init.end()))
        {
            Json::jsonmap_t jsonmap(stringpair_init.begin(),
                                    stringpair_init.end());
            Json json(jsonmap);
            Node* node = std::exchange(json.root, root);
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
    
    Options::Options(detail::listpair_t listpair)
        :Options()
        {
            OptionsList keys(std::move(listpair.first));
            OptionsList values(std::move(listpair.second));
            if (keys.count() != values.count()) { return; }
            std::size_t idx = 0,
                        max = keys.count();
            for (; idx < max; ++idx) {
                Options::set(keys[idx], static_cast<Json&&>(values[idx]));
            }
        }
    
    Options::Options(OptionsList&& keys, OptionsList&& values)
        :Options(std::make_pair(std::move(keys),
                                std::move(values)))
        {}
    
    Options::~Options() {}
    
    /// member swap
    void Options::swap(Options& other) noexcept {
        using std::swap;
        swap(cache, other.cache);
        swap(root,  other.root);
    }
    
    /// friend swap
    void swap(Options& lhs, Options& rhs) noexcept {
        lhs.swap(rhs);
    }
    
    Options& Options::operator=(stringpair_init_t stringpair_init) {
        Options(stringpair_init).swap(*this);
        return *this;
    }
    
    Options& Options::operator=(Json const& json) {
        Options(json).swap(*this);
        return *this;
    }
    
    Options& Options::operator=(Json&& json) noexcept {
        Node* node = std::exchange(json.root, root);
        (root = (node == nullptr ? &Node::null : node))->refcnt++;
        cache.clear();
        return *this;
    }
    
    Options& Options::operator=(Options const& other) {
        Options(other).swap(*this);
        return *this;
    }
    
    Options& Options::operator=(Options&& other) noexcept {
        Node* node = std::exchange(other.root, root);
        (root = (node == nullptr ? &Node::null : node))->refcnt++;
        cache = std::exchange(other.cache, cache);
        return *this;
    }
    
    bool Options::can_store() const noexcept { return true; }
    
    Options Options::parse(std::string const& str) {
        std::istringstream is(str);
        Options parsed(is);
        if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
        while (std::isspace(is.get()))
            /* skip */;
        if (is.eof()) { return parsed; }
        throw Json::parse_error("JSON im::Options format error", is);
    }
    
    bool Options::set(std::string const& key, std::string const& value) {
        if (key == STRINGNULL()) { return Options::del(key); }
        cache[key] = value;
        Json::set(key, value);
        return Json::has(key);
    }
    
    bool Options::set(std::string const& key, Json const& value) {
        if (key == STRINGNULL()) { return Options::del(key); }
        cache[key] = std::string(value);
        Json::set(key, value);
        return Json::has(key);
    }
    
    std::string& Options::get(std::string const& key) {
        if (cache.find(key) != cache.end()) { return cache[key]; }
        if (has(key)) {
            cache[key] = Json::cast<std::string>(key);
            return cache[key];
        }
        return STRINGNULL();
    }
    
    std::string const& Options::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) { return cache[key]; }
        if (has(key)) {
            cache[key] = Json::cast<std::string>(key);
            return cache[key];
        }
        return STRINGNULL();
    }
    
    bool Options::has(std::string const& key) const {
        return cache.find(key) != cache.end() || Json::has(key);
    }
    
    bool Options::del(std::string const& key) {
        if (cache.find(key) != cache.end()) { cache.erase(key); }
        return Json::remove(key);
    }
    
    std::size_t Options::count() const { return Json::size(); }
    stringvec_t Options::list() const { return Json::keys(); }
    
    std::size_t Options::count(std::regex const& pattern) const {
        stringvec_t keys = Options::list();
        return std::count_if(keys.begin(), keys.end(),
                         [&](std::string const& key) { return std::regex_search(key, pattern,
                                                              std::regex_constants::match_default); });
    }
    
    std::size_t Options::count(std::string const& prefix,
                               std::string const& separator) const {
        /// return the count of how many keys have this prefix:
        if (prefix.empty()) { return Options::prefixcount(separator[0]); }
        std::regex prefix_re("^" + prefix + separator, std::regex::extended);
        return Options::count(prefix_re);
    }
    
    std::size_t Options::prefixcount(std::string const& prefix,
                                     std::string const& separator) const {
        /// convenience function to call Options::count(prefix, separator)
        /// with “the default separator” which is ":" and/or ':', depending:
        if (prefix.empty()) { return Options::prefixcount(separator[0]); }
        std::regex prefix_re("^" + prefix + separator, std::regex::extended);
        return Options::count(prefix_re);
    }
    
    std::size_t Options::prefixcount(std::string::value_type sep) const {
        /// Count only the NON-prefixed keys in this version -- which means,
        /// count those keys that do not contain the separator of choice, yes!
        stringvec_t keys = Options::list();
        return std::count_if(keys.begin(), keys.end(),
                         [&](std::string const& key) { return bool(key.find(sep) == std::string::npos); });
    }
    
    prefixpair_t Options::prefixset(std::string const& separator) const {
        stringvec_t keys = Options::list();
        stringvec_t prefixvec;
        prefixset_t prefixes;
        
        /// fill a vector with prefixed keys:
        prefixvec.reserve(keys.size());
        std::copy_if(keys.begin(), keys.end(),
                     std::back_inserter(prefixvec),
                 [&](std::string const& s) { return bool(s.find(separator[0]) != std::string::npos); });
        
        /// cut the strings in the vector down to just the prefix:
        std::transform(prefixvec.begin(), prefixvec.end(),
                       prefixvec.begin(),
                   [&](std::string const& s) { return s.substr(0,
                                                      s.find_first_of(separator[0])); });
        
        /// uniquify the contents of the prefix string vector:
        std::sort(prefixvec.begin(), prefixvec.end());
        auto last = std::unique(prefixvec.begin(), prefixvec.end());
        prefixvec.erase(last, prefixvec.end());
        
        /// copy the unique prefixes into a set for output and return it:
        prefixes.reserve(prefixvec.size());
        std::copy(prefixvec.begin(),     prefixvec.end(),
                  std::inserter(prefixes, prefixes.end()));
        
        return std::make_pair(std::move(prefixes),
                              std::move(keys));
    }
    
    prefixgram_t Options::prefixgram(std::string const& separator) const {
        prefixpair_t prefixpair = Options::prefixset(separator);
        const prefixset_t prefixes = std::move(prefixpair.first);
        const stringvec_t keys = std::move(prefixpair.second);
        patternmap_t patterns;
        prefixgram_t prefixgram;
        
        /// fill a map with regex patterns, one pattern per prefix:
        patterns.reserve(prefixes.size());
        std::transform(prefixes.begin(),       prefixes.end(),
                       std::inserter(patterns, patterns.end()),
                   [&](std::string const& s) {
            std::regex re("^" + s + separator,  std::regex::extended);
            return std::make_pair(std::move(s),
                                  std::move(re));
        });
        
        /// count each pattern’s matches against the string vector
        /// containing a set of all keys, using a prefix histogram:
        prefixgram.reserve(patterns.size());
        std::for_each(patterns.begin(),
                      patterns.end(),
                  [&](auto const& kv) {
                      prefixgram[kv.first] = std::count_if(keys.begin(),
                                                           keys.end(),
                                                       [&](std::string const& s) {
                /// N.B. without that third argument, this call doesn’t compile;
                /// it errors out with a mass of ambiguous overload mismatches:
                return std::regex_search(s, kv.second,
                       std::regex_constants::match_default);
            });
        });
        
        /// return the prefix histogram (née “prefixgram”):
        return prefixgram;
    }
    
    ratios_t Options::ratios(std::string const& separator) const {
        stringvec_t keys = Json::keys();
                int total_count = static_cast<int>(keys.size());
                int unprefixed_count = std::count_if(keys.begin(),
                                                     keys.end(),
                                                 [&](std::string const& key) { return bool(key.find(separator[0]) == std::string::npos); });
                int prefixed_count = total_count - static_cast<int>(unprefixed_count);
             double total = static_cast<double>(total_count);
             double unprefixed = static_cast<double>(unprefixed_count);
             double prefixed = static_cast<double>(prefixed_count);
        if (total == 0.00) { return { -1.0, -1.0, 0.0 }; }
             double unprefixed_ratio = unprefixed == 0.00 ? 0.0 : (unprefixed / total);
             double prefixed_ratio = prefixed == 0.00 ? 0.0 : (prefixed / total);
        return std::make_tuple(unprefixed_ratio, prefixed_ratio,
                               unprefixed_count, prefixed_count,
                                                    total_count);
    }
    
    Options Options::subgroup(std::string const& name) const {
        if (Json::has(name)) {
            Json sg(Json::get(name));
            if (sg.type() == Type::OBJECT) {
                return Options(std::move(sg));
            }
        }
        return Options();
    }
    
    Options Options::subset(std::regex const& pattern,
                                         bool defix,
                           std::string const& replacement) const {
        stringvec_t keys = Options::list();
        stringvec_t pks;
        std::copy_if(keys.begin(), keys.end(),
                     std::back_inserter(pks),
                 [&](std::string const& key) { return std::regex_search(key, pattern,
                                                      std::regex_constants::match_default); });
        
        /// fill an ouput Options instance, per “defix”, with either:
        /// value copies for “de-fixed” keys (the default), or:
        /// value copies for identical keys, matching the original.
        Options out;
        if (pks.empty()) { return out; }
        if (defix) {
            for (std::string const& pk : pks) {
                out.set(std::regex_replace(pk, pattern, replacement),
                            Json::cast<std::string>(pk));   /// strip pattern from key string
            }
        } else {
            for (std::string const& pk : pks) {
                out.set(pk, Json::cast<std::string>(pk));   /// use key string as-is
            }
        }
        
        return out;
    }
    
    Options Options::subset(std::string const& prefix,
                                          bool defix,
                            std::string const& separator) const {
        std::regex prefix_re("^" + prefix + separator, std::regex::extended);
        return Options::subset(prefix_re, defix);
    }
    
    Options Options::replace(std::regex const& pattern,
                                          bool defix,
                            std::string const& replacement) const {
        /// fill an ouput Options instance, per “defix”, with either:
        /// value copies for “de-fixed” keys (the default), or:
        /// value copies for identical keys, matching the original.
        Options out;
        stringvec_t list = Options::list();
        if (list.empty()) { return out; }
        if (defix) {
            for (std::string const& pk : list) {
                out.set(std::regex_replace(pk, pattern, replacement),
                            Json::cast<std::string>(pk));   /// strip pattern from key string
            }
        } else {
            for (std::string const& pk : list) {
                out.set(pk, Json::cast<std::string>(pk));   /// use key string as-is
            }
        }
        
        return out;
    }
    
    Options Options::underscores_to_dashes() const {
        static const std::regex underscore_re("_", std::regex::extended);
        return Options::replace(underscore_re, true, "-");
    }
    
    Options Options::dashes_to_underscores() const {
        static const std::regex dash_re("-", std::regex::extended);
        return Options::replace(dash_re, true, "_");
    }
    
    OptionsList Options::keylist() const {
        OptionsList out(Options::list());
        return out;
    }
    
    OptionsList Options::valuelist() const {
        OptionsList out;
        for (std::string const& s : Options::list()) {
            out.append(Json::get(s));
        }
        return out;
    }
    
    detail::listpair_t Options::items() const {
        stringvec_t list = Options::list();
        OptionsList values;
        for (std::string const& s : list) {
            values.append(Json::get(s));
        }
        return std::make_pair(OptionsList(list),
                              std::move(values));
    }
    
    #pragma mark -
    #pragma mark im::get_optional_{string,cstring,int,bool} legacy methods
    
    std::string           get_optional_string(Options const& opts,
                                              std::string const& key) {
        return opts.cast<std::string>(key, "");
    }
    
    char const*           get_optional_cstring(Options const& opts,
                                               std::string const& key) {
        return opts.cast<char const*>(key, "");
    }
    
    int                   get_optional_int(Options const& opts,
                                           std::string const& key,
                                           int const default_value) {
        return opts.cast<int>(key, default_value);
    }
    
    bool                  get_optional_bool(Options const& opts,
                                            std::string const& key,
                                            bool const default_value) {
        return opts.cast<int>(key, static_cast<int>(default_value));
    }
    
} /* namespace im */

#pragma mark -
#pragma mark std::hash<…> specializations for im::OptionsList and im::Options

namespace std {
    
    using optlist_hasher_t = std::hash<im::OptionsList>;
    using optlist_arg_t = optlist_hasher_t::argument_type;
    using optlist_out_t = optlist_hasher_t::result_type;
    
    optlist_out_t optlist_hasher_t::operator()(optlist_arg_t const& os) const {
        return static_cast<optlist_out_t>(os.hash());
    }
    
    using opts_hasher_t = std::hash<im::Options>;
    using opts_arg_t = opts_hasher_t::argument_type;
    using opts_out_t = opts_hasher_t::result_type;
    
    opts_out_t opts_hasher_t::operator()(opts_arg_t const& o) const {
        return static_cast<opts_out_t>(o.hash());
    }
    
} /* namespace std */