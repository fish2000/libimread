#import <Foundation/Foundation.h>

#import "PDFStream.h"
#import "PDFEncryptionHandler.h"

extern NSString *PDFWrongMagicException;
extern NSString *PDFInvalidFormatException;
extern NSString *PDFParserException;

@interface PDFParser:NSObject
{
	CSHandle *fh;

	NSMutableDictionary *objdict;
	NSMutableArray *unresolved;

	NSDictionary *trailerdict;
	PDFEncryptionHandler *encryption;
}

+(PDFParser *)parserWithHandle:(CSHandle *)handle;
+(PDFParser *)parserForPath:(NSString *)path;

-(id)initWithHandle:(CSHandle *)handle;
-(void)dealloc;

-(BOOL)isEncrypted;
-(BOOL)needsPassword;
-(BOOL)setPassword:(NSString *)password;

-(NSDictionary *)objectDictionary;
-(NSDictionary *)trailerDictionary;
-(NSDictionary *)rootDictionary;
-(NSDictionary *)infoDictionary;

-(NSData *)permanentID;
-(NSData *)currentID;

-(NSDictionary *)pagesRoot;

-(PDFEncryptionHandler *)encryptionHandler;

-(void)parse;

-(NSDictionary *)parsePDFXref;
-(int)parseSimpleInteger;

-(id)parsePDFObject;

-(id)parsePDFTypeWithParent:(PDFObjectReference *)parent;
-(NSNull *)parsePDFNull;
-(NSNumber *)parsePDFBoolStartingWith:(int)c;
-(NSNumber *)parsePDFNumberStartingWith:(int)c;
-(NSString *)parsePDFWord;
-(NSString *)parsePDFStringWithParent:(PDFObjectReference *)parent;
-(NSData *)parsePDFHexStringStartingWith:(int)c parent:(PDFObjectReference *)parent;
-(NSArray *)parsePDFArrayWithParent:(PDFObjectReference *)parent;
-(NSDictionary *)parsePDFDictionaryWithParent:(PDFObjectReference *)parent;

-(void)resolveIndirectObjects;

-(void)_raiseParserException:(NSString *)error;

@end



@interface PDFString:NSObject <NSCopying>
{
	NSData *data;
	PDFObjectReference *ref;
	PDFParser *parser;
}

-(id)initWithData:(NSData *)bytes parent:(PDFObjectReference *)parent parser:(PDFParser *)owner;
-(void)dealloc;

-(NSData *)data;
-(PDFObjectReference *)reference;
-(NSData *)rawData;
-(NSString *)string;

-(BOOL)isEqual:(id)other;
-(unsigned)hash;

-(id)copyWithZone:(NSZone *)zone;

-(NSString *)description;

@end



@interface PDFObjectReference:NSObject <NSCopying>
{
	int num,gen;
}

+(PDFObjectReference *)referenceWithNumber:(int)objnum generation:(int)objgen;
+(PDFObjectReference *)referenceWithNumberObject:(NSNumber *)objnum generationObject:(NSNumber *)objgen;

-(id)initWithNumber:(int)objnum generation:(int)objgen;

-(int)number;
-(int)generation;

-(BOOL)isEqual:(id)other;
-(unsigned)hash;

-(id)copyWithZone:(NSZone *)zone;

-(NSString *)description;

@end
