#ifndef INICPP_STRING_UTILS_H
#define INICPP_STRING_UTILS_H

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#include "exception.h"
#include "types.h"

namespace inicpp {
    /**
     * Namespace which contains methods for working with strings, including
     * basic string operations and inicpp specific parsing.
     */
    namespace string_utils {
        /**
         * Trim whitespaces from start of given string.
         * @param str processed string
         * @return newly created instance of string
         */
        std::string left_trim(std::string const& str);
        /**
         * Trim whitespaces from end of given string.
         * @param str processed string
         * @return newly created instance of string
         */
        std::string right_trim(std::string const& str);
        /**
         * Trim whitespaces from start and end of given string.
         * @param str processed string
         * @return newly created instance of string
         */
        std::string trim(std::string const& str);
        /**
         * In @a haystack find any occurence of needle.
         * @param haystack string in which search is executed
         * @param needle string which is searched for
         * @return true if at least one @a needle was found, false otherwise
         */
        bool find_needle(std::string const& haystack, std::string const& needle);
        /**
         * Tries to find out if given @a str starts with @a search_str.
         * @param str searched string
         * @param search_str string which is searched for
         * @return true if given string starts with @a search_str
         */
        bool starts_with(std::string const& str, std::string const& search_str);
        /**
         * Tries to find out if given @a str ends with @a search_str.
         * @param str searched string
         * @param search_str string which is searched for
         * @return true if given string starts with @a search_str
         */
        bool ends_with(std::string const& str, std::string const& search_str);
        /**
         * Split given string with given delimiter.
         * @param str text which will be splitted
         * @param delim delimiter
         * @return array of newly created substrings
         */
        std::vector<std::string> split(std::string const& str, char delim);

        /**
         * Function for parsing string input value to strongly typed one
         * @param value Value to be parsed
         * @param option_name Option name from this value - will be in exception text if thrown
         * @result Value parsed to proper (ReturnType) type
         * @throws invalid_type_exception if such cast cannot be made
         */
        template <typename ReturnType>
        ReturnType parse_string(std::string const& value, std::string const& option_name) {
            throw invalid_type_exception("Invalid option type");
        }
        /**
         * Specialization for string type, which doesn't need to be explicitely parsed.
         */
        template <>
        string_ini_t parse_string<string_ini_t>(std::string const& value, std::string const&);
        /**
         * Parse string to boolean value.
         * @param value Value to be parsed
         * @param option_name Option name from this value - will be in exception text if thrown
         * @return parsed value with correct type
         * @throws invalid_type_exception if string cannot be parsed
         */
        template <>
        boolean_ini_t parse_string<boolean_ini_t>(std::string const& value,
                                                  std::string const& option_name);
        /**
         * Parse string to enum value.
         * @param value Value to be parsed
         * @param option_name Option name from this value - will be in exception text if thrown
         * @return parsed value with correct type
         * @throws invalid_type_exception if string cannot be parsed
         */
        template <>
        enum_ini_t parse_string<enum_ini_t>(std::string const& value,
                                            std::string const& option_name);
        /**
         * Parse string to float value.
         * @param value Value to be parsed
         * @param option_name Option name from this value - will be in exception text if thrown
         * @return parsed value with correct type
         * @throws invalid_type_exception if string cannot be parsed
         */
        template <>
        float_ini_t parse_string<float_ini_t>(std::string const& value,
                                              std::string const& option_name);
        /**
         * Parse string to signed value.
         * @param value Value to be parsed
         * @param option_name Option name from this value - will be in exception text if thrown
         * @return parsed value with correct type
         * @throws invalid_type_exception if string cannot be parsed
         */
        template <>
        signed_ini_t parse_string<signed_ini_t>(std::string const& value,
                                                std::string const& option_name);
        /**
         * Parse string to unsigned value.
         * @param value Value to be parsed
         * @param option_name Option name from this value - will be in exception text if thrown
         * @return parsed value with correct type
         * @throws invalid_type_exception if string cannot be parsed
         */
        template <>
        unsigned_ini_t parse_string<unsigned_ini_t>(std::string const& value,
                                                    std::string const& option_name);
    } // namespace string_utils

    /** Internal namespace to hide to_string methods. */
    namespace inistd {
        /** Standard std::to_string method for all types */
        using std::to_string;
        /** Custom to_string method for enum_ini_t type */
        std::string to_string(enum_ini_t const& value);
    }; // namespace inistd
} // namespace inicpp

#endif // INICPP_STRING_UTILS_H
