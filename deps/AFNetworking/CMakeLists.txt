# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers
cmake_minimum_required(VERSION 3.1)
project(AFNetworking)

set(AFNetworking_VERSION_MAJOR "3")
set(AFNetworking_VERSION_MINOR "0")
set(AFNetworking_VERSION_PATCH "1")

find_path(FOUNDATION_INCLUDE_DIR Foundation/Foundation.h)
find_library(FOUNDATION_LIBRARY Foundation)
find_path(COREFOUNDATION_INCLUDE_DIR CoreFoundation/CoreFoundation.h)
find_library(COREFOUNDATION_LIBRARY CoreFoundation)
find_path(APPKIT_INCLUDE_DIR AppKit/AppKit.h)
find_library(APPKIT_LIBRARY AppKit)
find_path(SYSTEMCONFIGURATION_INCLUDE_DIR SystemConfiguration/SystemConfiguration.h)
find_library(SYSTEMCONFIGURATION_LIBRARY SystemConfiguration)
find_path(SECURITY_INCLUDE_DIR Security/Security.h)
find_library(SECURITY_LIBRARY Security)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    ${FOUNDATION_INCLUDE_DIR}
    ${COREFOUNDATION_INCLUDE_DIR}
    ${APPKIT_INCLUDE_DIR}
    ${SYSTEMCONFIGURATION_INCLUDE_DIR}
    ${SECURITY_INCLUDE_DIR})

add_definitions(
    -Wall -O3
    -fstack-protector-all
    -fobjc-arc
    -fobjc-abi-version=3
    -std=c99
    -mtune=native
    -x objective-c)

set(EXTRA_LIBS ${EXTRA_LIBS}
    ${FOUNDATION_LIBRARY}
    ${COREFOUNDATION_LIBRARY}
    ${APPKIT_LIBRARY}
    ${SYSTEMCONFIGURATION_LIBRARY}
    ${SECURITY_LIBRARY})

set(srcs
    "AFNetworking/AFHTTPSessionManager.m"
    "AFNetworking/AFNetworkReachabilityManager.m"
    "AFNetworking/AFSecurityPolicy.m"
    "AFNetworking/AFURLRequestSerialization.m"
    "AFNetworking/AFURLResponseSerialization.m"
    "AFNetworking/AFURLSessionManager.m")

set(hdrs
    "AFNetworking/AFHTTPSessionManager.h"
    "AFNetworking/AFNetworking.h"
    "AFNetworking/AFNetworkReachabilityManager.h"
    "AFNetworking/AFSecurityPolicy.h"
    "AFNetworking/AFURLRequestSerialization.h"
    "AFNetworking/AFURLResponseSerialization.h"
    "AFNetworking/AFURLSessionManager.h")

add_library(AFNetworking SHARED ${srcs} ${hdrs})

set_target_properties(AFNetworking
    PROPERTIES OUTPUT_NAME "AFNetworking")
target_link_libraries(AFNetworking ${EXTRA_LIBS})

install(TARGETS AFNetworking DESTINATION lib)
install(DIRECTORY ./ DESTINATION include/AFNetworking
    FILES_MATCHING PATTERN "*.h")
