/*
 * Copyright (c) 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015
 *   Jonathan Schleifer <js@webkeks.org>
 *
 * All rights reserved.
 *
 * This file is part of ObjFW. It may be distributed under the terms of the
 * Q Public License 1.0, which can be found in the file LICENSE.QPL included in
 * the packaging of this file.
 *
 * Alternatively, it may be distributed under the terms of the GNU General
 * Public License, either version 2 or 3, which can be found in the file
 * LICENSE.GPLv2 or LICENSE.GPLv3 respectively included in the packaging of this
 * file.
 */

#import "OFStreamSocket.h"

#import "socket.h"

OF_ASSUME_NONNULL_BEGIN

/*! @file */

@class OFTCPSocket;
@class OFString;

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief A block which is called when the socket connected.
 *
 * @param socket The socket which connected
 * @param exception An exception which occurred while connecting the socket or
 *		    nil on success
 */
typedef void (^of_tcp_socket_async_connect_block_t)(OFTCPSocket *socket,
    OFException *OF_NULLABLE exception);

/*!
 * @brief A block which is called when the socket accepted a connection.
 *
 * @param socket The socket which accepted the connection
 * @param acceptedSocket The socket which has been accepted
 * @param exception An exception which occurred while accepting the socket or
 *		    nil on success
 * @return A bool whether the same block should be used for the next incoming
 *	   connection
 */
typedef bool (^of_tcp_socket_async_accept_block_t)(OFTCPSocket *socket,
    OFTCPSocket *acceptedSocket, OFException *OF_NULLABLE exception);
#endif

/*!
 * @class OFTCPSocket OFTCPSocket.h ObjFW/OFTCPSocket.h
 *
 * @brief A class which provides functions to create and use TCP sockets.
 *
 * To connect to a server, create a socket and connect it.
 * To create a server, create a socket, bind it and listen on it.
 */
@interface OFTCPSocket: OFStreamSocket
{
	bool _listening;
	struct sockaddr *_address;
	socklen_t _addressLength;
	OFString *_SOCKS5Host;
	uint16_t _SOCKS5Port;
}

#ifdef OF_HAVE_PROPERTIES
@property (readonly, getter=isListening) bool listening;
@property OF_NULLABLE_PROPERTY (copy) OFString *SOCKS5Host;
@property uint16_t SOCKS5Port;
@property (getter=isKeepAliveEnabled) bool keepAliveEnabled;
@property (getter=isTCPNoDelayEnabled) bool TCPNoDelayEnabled;
#endif

/*!
 * @brief Sets the global SOCKS5 proxy host to use when creating a new socket
 *
 * @param SOCKS5Host The host to use as a SOCKS5 proxy when creating a new
 *		     socket
 */
+ (void)setSOCKS5Host: (nullable OFString*)SOCKS5Host;

/*!
 * @brief Returns the host to use as a SOCKS5 proxy when creating a new socket
 *
 * @return The host to use as a SOCKS5 proxy when creating a new socket
 */
+ (nullable OFString*)SOCKS5Host;

/*!
 * @brief Sets the global SOCKS5 proxy port to use when creating a new socket
 *
 * @param SOCKS5Port The port to use as a SOCKS5 proxy when creating a new socket
 */
+ (void)setSOCKS5Port: (uint16_t)SOCKS5Port;

/*!
 * @brief Returns the port to use as a SOCKS5 proxy when creating a new socket
 *
 * @return The port to use as a SOCKS5 proxy when creating a new socket
 */
+ (uint16_t)SOCKS5Port;

/*!
 * @brief Sets the host to use as a SOCKS5 proxy.
 *
 * @param host The host to use as a SOCKS5 proxy
 */
- (void)setSOCKS5Host: (nullable OFString*)host;

/*!
 * @brief Returns the host to use as a SOCKS5 proxy.
 *
 * @return The host to use as a SOCKS5 proxy
 */
- (nullable OFString*)SOCKS5Host;

/*!
 * @brief Sets the port to use on the SOCKS5 proxy.
 *
 * The default port is 1080.
 *
 * @param port The port to use on the SOCKS5 proxy
 */
- (void)setSOCKS5Port: (uint16_t)port;

/*!
 * @brief Returns the port to use on the SOCKS5 proxy.
 *
 * @return The port to use on the SOCKS5 proxy
 */
- (uint16_t)SOCKS5Port;

/*!
 * @brief Connect the OFTCPSocket to the specified destination.
 *
 * @param host The host to connect to
 * @param port The port on the host to connect to
 */
- (void)connectToHost: (OFString*)host
		 port: (uint16_t)port;

#ifdef OF_HAVE_THREADS
/*!
 * @brief Asyncronously connect the OFTCPSocket to the specified destination.
 *
 * @param host The host to connect to
 * @param port The port on the host to connect to
 * @param target The target on which to call the selector once the connection
 *		 has been established
 * @param selector The selector to call on the target. The signature must be
 *		   `void (OFTCPSocket *socket, OFException *exception)`.
 */
- (void)asyncConnectToHost: (OFString*)host
		      port: (uint16_t)port
		    target: (id)target
		  selector: (SEL)selector;

# ifdef OF_HAVE_BLOCKS
/*!
 * @brief Asyncronously connect the OFTCPSocket to the specified destination.
 *
 * @param host The host to connect to
 * @param port The port on the host to connect to
 * @param block The block to execute once the connection has been established
 */
- (void)asyncConnectToHost: (OFString*)host
		      port: (uint16_t)port
		     block: (of_tcp_socket_async_connect_block_t)block;
# endif
#endif

/*!
 * @brief Bind the socket to the specified host and port.
 *
 * @param host The host to bind to. Use `@"0.0.0.0"` for IPv4 or `@"::"` for
 *	       IPv6 to bind to all.
 * @param port The port to bind to. If the port is 0, an unused port will be
 *	       chosen, which can be obtained using the return value.
 * @return The port the socket was bound to
 */
- (uint16_t)bindToHost: (OFString*)host
		  port: (uint16_t)port;

/*!
 * @brief Listen on the socket.
 *
 * @param backLog Maximum length for the queue of pending connections.
 */
- (void)listenWithBackLog: (int)backLog;

/*!
 * @brief Listen on the socket.
 */
- (void)listen;

/*!
 * @brief Accept an incoming connection.
 *
 * @return An autoreleased OFTCPSocket for the accepted connection.
 */
- (instancetype)accept;

/*!
 * @brief Asyncronously accept an incoming connection.
 *
 * @param target The target on which to execute the selector when a new
 *		 connection has been accepted. The method returns whether the
 *		 next incoming connection should be accepted by the specified
 *		 block as well.
 * @param selector The selector to call on the target. The signature must be
 *		   `bool (OFTCPSocket *socket, OFTCPSocket *acceptedSocket,
 *		   OFException *exception)`.
 */
- (void)asyncAcceptWithTarget: (id)target
		     selector: (SEL)selector;

#ifdef OF_HAVE_BLOCKS
/*!
 * @brief Asyncronously accept an incoming connection.
 *
 * @param block The block to execute when a new connection has been accepted.
 *		Returns whether the next incoming connection should be accepted
 *		by the specified block as well.
 */
- (void)asyncAcceptWithBlock: (of_tcp_socket_async_accept_block_t)block;
#endif

/*!
 * @brief Returns the remote address of the socket.
 *
 * Only works with accepted sockets!
 *
 * @return The remote address as a string
 */
- (nullable OFString*)remoteAddress;

/*!
 * @brief Returns whether the socket is a listening socket.
 *
 * @return Whether the socket is a listening socket
 */
- (bool)isListening;

/*!
 * @brief Enable or disable keep alive for the connection.
 *
 * @param enabled Whether to enable or disable keep alives for the connection
 */
- (void)setKeepAliveEnabled: (bool)enabled;

/*!
 * @brief Returns whether keep alive is enabled for the connection.
 *
 * @return Whether keep alives are enabled for the connection
 */
- (bool)isKeepAliveEnabled;

/*!
 * @brief Enable or disable TCP_NODELAY for the connection.
 *
 * @param enabled Whether to enable or disable TCP_NODELAY for the connection
 */
- (void)setTCPNoDelayEnabled: (bool)enabled;

/*!
 * @brief Returns whether TCP_NODELAY is enabled for the connection.
 *
 * @return Whether TCP_NODELAY is enabled for the connection
 */
- (bool)isTCPNoDelayEnabled;
@end

#ifdef __cplusplus
extern "C" {
#endif
extern Class OF_NULLABLE of_tls_socket_class;
#ifdef __cplusplus
}
#endif

OF_ASSUME_NONNULL_END
