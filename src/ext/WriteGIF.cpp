
#include <libimread/ext/WriteGIF.h>
#include <libimread/ext/fmemopen.hh>
#include <libimread/errors.hh>

using im::byte;

/**************************************************
        header
        //logical screen =
            logical screen descriptor
            [global color table]
        //data =
            //graphic block =
                [graphic control extension]
                //graphic-rendering block =
                    //table-based image =
                        image descriptor
                        [local color table]
                        image data
                    plain text extension
            //special purpose block =
                application extension
                comment extension
        trailer
**************************************************/

#define VERBOSE 0

#if (VERBOSE > 0)
#define MAYBE(...) YES(__VA_ARGS__)
#else
#define MAYBE(...)
#endif



namespace {
    
    void swap(unsigned char *a, unsigned char *b) {
        for (int i = 0; i < 3; i++) {
            unsigned char x = a[i];
            a[i] = b[i];
            b[i] = x;
        }
    }
    
    void sortColorsByAxis(unsigned char *array,
                          int count, int axis) {
        if (count < 2) { return; }
        unsigned char pivot = (array + (count/2)*3)[axis];
        unsigned char *left = array, *right = array + (count-1)*3;
        while (left < right) {
            while (left[axis] < pivot) { left += 3; }
            while (right[axis] > pivot) { right -= 3; }
            if (left < right) {
                swap(left, right);
                left += 3;
                right -= 3;
            }
        }
        int leftCount = (left - array)/3;
        int rightCount = count - leftCount;
        if (leftCount > 1) { sortColorsByAxis(array, leftCount, axis); }
        if (rightCount > 1) { sortColorsByAxis(array+leftCount*3, rightCount, axis); }
    }

    static int nearestIndexInPalette(unsigned char *palette,
                                     int paletteSize, unsigned char *rgb) {
        int bestIndex = 0, bestDist = 0;
        for (int i = 0; i < paletteSize; i++) {
            unsigned char *p = palette + i * 3;
            int dr = p[0] - rgb[0], dg = p[1] - rgb[1], db = p[2] - rgb[2];
            int d = dr * dr + dg * dg + db * db;
            if (d == 0) { return i; }
            if (bestDist == 0 || d < bestDist) {
                bestIndex = i;
                bestDist = d;
            }
        }
        return bestIndex;
    }

    static void indexizeImageFromPaletteFuzzy(
        int Width, int Height, unsigned char *rgbImage, unsigned char *indexImage, 
        unsigned char *palette, int paletteSize) {
        for (int i = 0; i < Width * Height; i++) {
            unsigned char *rgb = rgbImage + 3 * i;
            indexImage[i] = nearestIndexInPalette(palette, paletteSize, rgb);
        }
    }

    static void writeTransparentPixelsWhereNotDifferent(
        unsigned char *prevImage, unsigned char *thisImage, unsigned char *outImage,
        int ImageWidth, int ImageHeight, int TranspValue) {
        int count = 0;
        for (int i = 0; i < ImageWidth * ImageHeight; i++) {
            if (thisImage[i] == prevImage[i]) {
                outImage[i] = TranspValue;
                ++count;
            } else {
                outImage[i] = thisImage[i];
            }
        }
        MAYBE(FF(" - %d%% transparent", count * 100 / (ImageWidth*ImageHeight)));
    }

    void calculatePossibleCrop(int width, int height,
                               unsigned char *indexImage,
                               unsigned char TranspColorIndex,
                               int &left, int &right, int &top, int &bottom) {
        left = width, right = 0, top = height, bottom = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                unsigned char v = *(indexImage++);
                if (v != TranspColorIndex) {
                    if (x < left) { left = x; }
                    if (right < x) { right = x; }
                    if (y < top) { top = y; }
                    if (bottom < y) { bottom = y; }
                }
            }
        }
    }
    
    struct BlockWriter {
        FILE *f;
        char bytes[255];
        char curBits;
        int byteCount;
        int curBitMask;
        int totalBytesWritten;
    
        BlockWriter(FILE *f)
            :f(f)
            ,curBits(0), byteCount(0)
            ,curBitMask(1), totalBytesWritten(0)
            {}
        
        void finish() {
            if (curBitMask > 1) { writeByte(); }
            if (byteCount > 0) { writeBlock(); }
            fputc(0, f); /// block terminator
        }
        
        void writeBlock() {
            if (f) {
                fputc(byteCount, f);
                fwrite(bytes, byteCount, 1, f);
            }
            byteCount = 0;
        }
        
        void writeByte() {
            totalBytesWritten++;
            bytes[byteCount] = curBits;
            byteCount++;
            if (byteCount >= 255) { writeBlock(); }
            curBitMask = 1;
            curBits = 0;
        }
        
        void writeBit(int bit) {
            if (bit & 1) {
                curBits |= curBitMask;
            } else {
                /// nothing here,
                /// because curBits should have been zeroed
            }
            curBitMask <<= 1;
            if (curBitMask > 0x80) { writeByte(); }
        }
    
        void writeBitArray(int bits, int bitCount) {
            while (bitCount-- > 0) {
                writeBit(bits & 1);
                bits >>= 1;
            }
        }
    
    };
    
    struct TableEntry {
        int length, index;
        TableEntry *after[256];
        TableEntry() {
            for (int i = 0; i < 256; i++) { after[i] = NULL; }
        }
        
        void deleteChildren() {
            for (int i = 0; i < 256; i++) {
                if (after[i]) {
                    after[i]->deleteChildren();
                    delete after[i];
                    after[i] = NULL;
                }
            }
        }
        
        TableEntry *findLongestString(unsigned char *input, int inputSize) {
            if (inputSize > 0) {
                if (after[input[0]]) {
                    return after[input[0]]->findLongestString(input+1, inputSize-1);
                } else { return this; }
            } else { return this; }
        }
        
        void resetTable() {
            for (int i = 0; i < 256; i++) {
                if (after[i]) {
                    after[i]->deleteChildren();
                } else {
                    after[i] = new TableEntry();
                    after[i]->length = 1;
                    after[i]->index = i;
                }
            }
            length = 256 + 2; /// reserve "clear code" and "end of information"
        }
    
        void insertAfter(TableEntry *entry, unsigned char code) {
            if (entry->after[code]) {
                WTF("WTF?");
            } else {
                entry->after[code] = new TableEntry;
                entry->after[code]->length = entry->length + 1;
                entry->after[code]->index = length++;
            }
        }
    
    };
    
    void encode(BlockWriter& output, unsigned char *input, int inputSize,
                const int InitCodeSize, const int MaxCodeSize) {
        const int ClearCode = (1 << InitCodeSize);
        const int EndOfInformation = ClearCode + 1;
        int codeSize = InitCodeSize;
        TableEntry table;
        table.resetTable();
        output.writeBitArray(ClearCode, codeSize+1);
        while (inputSize > 0) {
            TableEntry *entry = table.findLongestString(input, inputSize);
            if (!entry) { WTF("WTF-2"); }
            
            output.writeBitArray(entry->index, codeSize+1);
            input += entry->length;
            inputSize -= entry->length;
            
            if (inputSize > 0) {
                table.insertAfter(entry, input[0]);
                if ((1 << (codeSize+1)) < table.length) {
                    ++codeSize;
                    //WTF(FF("code size %d\n", codeSize));
                }
                if ((codeSize + 1 >= MaxCodeSize) && ((1 << (codeSize+1)) <= table.length)) {
                    //WTF("reset table");
                    output.writeBitArray(ClearCode, codeSize+1);
                    table.resetTable();
                    codeSize = InitCodeSize;
                }
            }
        }
        output.writeBitArray(EndOfInformation, codeSize+1);
        table.deleteChildren();
    }
    
} /// end of private namespace


namespace gif {
    
    struct Frame {
        Frame *next;
        unsigned char *rgbImage;
        unsigned char *indexImage;
        int delay; ///* 1/100 sec
    };
    
    struct GIF {
        int width, height;
        Frame *frames, *lastFrame;
        int frameDelay;
        unsigned char *palette;
        int paletteSize;
    };
    
    void dispose(GIF *gif) {
        Frame *f = gif->frames;
        while (f) {
            Frame *next = f->next;
            if (f->indexImage) { delete[] f->indexImage; }
            if (f->rgbImage) { delete[] f->rgbImage; }
            delete f; f = next;
        }
        if (gif->palette) { delete[] gif->palette; }
        delete gif;
    }
    
    static const int TranspColorIndex = 255; /// arbitrary, [0..255]
    
    bool isAnimated(GIF *gif) { return (gif->frames && gif->frames->next); }
    
    GIF *newGIF(int delay) {
        GIF *gif = new GIF;
        gif->width = 0, gif->height = 0;
        gif->frames = NULL;
        gif->lastFrame = NULL;
        gif->frameDelay = delay;
        gif->palette = NULL;
        gif->paletteSize = 0;
        return gif;
    }
    
    bool setHasColor(char set[256*256*256/8], unsigned char *rgb) {
        int i = int(rgb[0]) * 256 * 256 + int(rgb[1]) * 256 + int(rgb[2]);
        return ((set[i/8] & (1 << (i%8))) != 0);
    }
    
    void addColorToSet(char set[256*256*256/8], unsigned char *rgb) {
        int i = int(rgb[0]) * 256 * 256 + int(rgb[1]) * 256 + int(rgb[2]);
        set[i/8] |= (1 << (i%8));
    }
    
    void calculatePaletteByMedianCut(GIF *gif) {
        MAYBE("Caculating palette by median cut");
        unsigned char *uniqueColorArray = NULL;
        int uniqueColorCount = 0, idx = 0;
        
        static char colorBitSet[256*256*256/8];
        {
            //printf("Calculating unique color count [");
            idx = 0;
            std::memset(colorBitSet, 0, 256*256*256/8);
            for (Frame *frame = gif->frames; frame != NULL; frame = frame->next) {
                //printf("*");
                idx++;
                unsigned char *end = frame->rgbImage + gif->width * gif->height * 3;
                for (unsigned char *rgb = frame->rgbImage; rgb < end; rgb += 3) {
                    if (!setHasColor(colorBitSet, rgb)) {
                        addColorToSet(colorBitSet, rgb);
                        uniqueColorCount++;
                    }
                }
            }
            //printf("]\nUnique color count %d\n", uniqueColorCount);
            std::unique_ptr<char[]> asterisks = std::make_unique<char[]>(idx+1);
            std::memset(asterisks.get(), '*', idx);
            MAYBE(
                FF("Calculating unique color count [%s]", asterisks.get()),
                FF("Unique color count: %d", uniqueColorCount)
            );
        }
        
        uniqueColorArray = new unsigned char[uniqueColorCount * 3];
        {
            //printf("Filling unique color array [");
            idx = 0;
            memset(colorBitSet, 0, 256*256*256/8);
            unsigned char *afterLastUnique = uniqueColorArray + uniqueColorCount * 3;
            unsigned char *u = uniqueColorArray;
            for (Frame *frame = gif->frames; frame != NULL; frame = frame->next) {
                //printf("*");
                idx++;
                if (u >= afterLastUnique) { break; }
                unsigned char *end = frame->rgbImage + gif->width * gif->height * 3;
                for (unsigned char *rgb = frame->rgbImage; rgb < end; rgb += 3) {
                    if (u >= afterLastUnique) { break; }
                    if (!setHasColor(colorBitSet, rgb)) {
                        addColorToSet(colorBitSet, rgb);
                        u[0] = rgb[0], u[1] = rgb[1], u[2] = rgb[2];
                        u += 3;
                    }
                }
            }
            //printf("]\n");
            std::unique_ptr<char[]> asterisks = std::make_unique<char[]>(idx+1);
            std::memset(asterisks.get(), '*', idx);
            MAYBE(
                FF("Filling unique color array [%s]", asterisks.get())
            );
            
        }
        
        struct ColorBox {
            char splitAxis;
            char splitValue;
            ColorBox *child[2];
            int dim[3];
            unsigned char *colors;
            int colorCount;
            
            bool isLeaf() { return child[0] == NULL && child[1] == NULL; }
            
            void calcDim() {
                int minDim[3] = { 255, 255, 255 }, maxDim[3] = { 0, 0, 0 };
                for (int i = 0; i < colorCount; i++) {
                    unsigned char *rgb = colors + i * 3;
                    for (int a = 0; a < 3; a++) {
                        if (rgb[a] < minDim[a]) { minDim[a] = rgb[a]; }
                        if (maxDim[a] < rgb[a]) { maxDim[a] = rgb[a]; }
                    }
                }
                for (int a = 0; a < 3; a++) { dim[a] = maxDim[a] - minDim[a]; }
                //printf("minDim(%d,%d,%d), maxDim(%d,%d,%d), dim(%d,%d,%d)\n", minDim[0],minDim[1],minDim[2], maxDim[0],maxDim[1],maxDim[2], dim[0], dim[1], dim[2]);
            }
            
            void calcColor(unsigned char *rgbOut) {
                int r = 0, g = 0, b = 0;
                for (int i = 0; i < colorCount; i++) {
                    r += (colors + i * 3)[0];
                    g += (colors + i * 3)[1];
                    b += (colors + i * 3)[2];
                }
                rgbOut[0] = r / colorCount;
                rgbOut[1] = g / colorCount;
                rgbOut[2] = b / colorCount;
            }
        };
        
        static const int ColorBoxArraySize = 512;
        static ColorBox colorBoxArray[ColorBoxArraySize];
        int colorBoxCount = 0;
        int leafBoxCount = 0;
        
        {
            //printf("Creating color boxes [");
            memset(&colorBoxArray, 0, sizeof(ColorBox) * 512);
            colorBoxArray[0].colors = uniqueColorArray;
            colorBoxArray[0].colorCount = uniqueColorCount;
            colorBoxArray[0].calcDim();
            colorBoxCount = 1;
            leafBoxCount = 1;
            
            idx = 0;
            while (leafBoxCount < 255) {
                //printf("*");
                idx++;
                int maxDimAxis = 0;
                int maxDim = 0;
                ColorBox *maxDimBox = NULL;
                for (int i = 0; i < colorBoxCount; i++) {
                    ColorBox *box = colorBoxArray + i;
                    if (!box->isLeaf()) { continue; }
                    if (box->colorCount < 2) { continue; }
                    
                    for (int axis = 0; axis < 3; axis++) {
                        if (box->dim[axis] > maxDim || maxDimBox == NULL) {
                            maxDim = box->dim[axis];
                            maxDimAxis = axis;
                            maxDimBox = box;
                        }
                    }
                }
                if (maxDim < 2) { break; }
                //printf("maxDim %d, maxDimAxis %d\n", maxDim, maxDimAxis);
                
                if (colorBoxCount + 2 <= ColorBoxArraySize) {
                    maxDimBox->splitAxis = maxDimAxis;
                    //maxDimBox->splitValue = ???;
                    sortColorsByAxis(maxDimBox->colors, maxDimBox->colorCount, maxDimAxis);
                    ColorBox *L = maxDimBox->child[0] = &colorBoxArray[colorBoxCount++];
                    ColorBox *R = maxDimBox->child[1] = &colorBoxArray[colorBoxCount++];
                    //printf("Sorted array by axis\n"); for (int i = 0; i <maxDimBox->colorCount; i++) printf("%d, ", (maxDimBox->colors+i*3)[maxDimAxis]); printf("\n");
                    
                    L->colors = maxDimBox->colors;
                    L->colorCount = maxDimBox->colorCount / 2;
                    R->colors = maxDimBox->colors + L->colorCount * 3;
                    R->colorCount = maxDimBox->colorCount - L->colorCount;
                    L->calcDim();
                    R->calcDim();
                    leafBoxCount++;
                } else {
                    WTF("Error: insufficient color box array size");
                    break;
                }
            }
            
            //printf("]\n");
            //printf("Total box count %d, leaf box count %d\n", colorBoxCount, leafBoxCount);
            std::unique_ptr<char[]> asterisks = std::make_unique<char[]>(idx+1);
            std::memset(asterisks.get(), '*', idx);
            MAYBE(
                FF("Creating color boxes [%s]", asterisks.get()),
                FF("Total box count: %d", colorBoxCount),
                FF(" Leaf box count: %d", leafBoxCount)
            );
            
            
            {
                MAYBE("Calculating palette from boxes");
                gif->paletteSize = leafBoxCount;
                gif->palette = new unsigned char[256*3];
                unsigned char *rgbPal = gif->palette;
                for (int i = 0; i < colorBoxCount; i++) {
                    if (colorBoxArray[i].isLeaf()) {
                        colorBoxArray[i].calcColor(rgbPal);
                        rgbPal += 3;
                    }
                }
                //printf("Indexizing frames [");
                idx = 0;
                for (Frame *frame = gif->frames; frame != NULL; frame = frame->next) {
                    //printf("*");
                    idx++;
                    frame->indexImage = new unsigned char[gif->width * gif->height];
                    indexizeImageFromPaletteFuzzy(gif->width, gif->height, frame->rgbImage, frame->indexImage, gif->palette, gif->paletteSize);
                }
                //printf("]\n");
                std::unique_ptr<char[]> asterisks = std::make_unique<char[]>(idx+1);
                std::memset(asterisks.get(), '*', idx);
                MAYBE(
                    FF("Indexizing frames [%s]", asterisks.get())
                );
                
            }
        }
        delete[] uniqueColorArray;
    }
    
    void addFrame(GIF *gif, int W, int H, unsigned char *rgbImage, int delay) {
        Frame *f = new Frame;
        f->delay = (delay < 0) ? gif->frameDelay : delay;
        f->indexImage = NULL;
        f->rgbImage = new unsigned char[W*H*3];
        std::memcpy(f->rgbImage, rgbImage, W*H*3);
        f->next = NULL;
        if (gif->lastFrame) {
            gif->lastFrame->next = f;
        } else {
            gif->frames = f;
        }
        gif->lastFrame = f;
        if (gif->width && gif->height) {
            if (gif->width != W || gif->height != H) {
                WTF("Frame width/height differ from GIF's!\n");
            }
        } else {
            /// initialize the internal structs' width and height
            gif->width = W;
            gif->height = H;
        }
    }
    
    /// this used to write to a filehandle -- now it does so internally
    /// and captures the results using memory::sink for return
    std::vector<byte> write(GIF *gif) {
        if (!gif->frames) {
            //WTF("GIF incomplete\n"); return;
            throw im::CannotWriteError("Incomplete GIF passed to gif::write()");
        }
        
        {
            /// calculate global palette
            calculatePaletteByMedianCut(gif);
        }
        
        // const char *filename
        // if (!filename) {
        //     static char defaultFilename[256] = "test.gif";
        //     snprintf(defaultFilename, 256, "%d.gif", int(time(0)));
        //     filename = defaultFilename;
        // }
        // FILE *f = fopen(filename, "wb");
        // if (!f) { printf("Failed open for writing %s\n", filename); return; }
        // printf("Writing %s...\n", filename);
        
        /// count frames
        int framecount = 0;
        for (Frame *frame = gif->frames, *prevFrame = NULL;
             frame != NULL;
             prevFrame = frame, frame = frame->next, ++framecount) {}
        
        /// allocate way-too-large buffer vector (which we resize before returning)
        int membufsize = gif->width * gif->height * 3 * framecount;
        std::vector<byte> overflow(membufsize);
        std::unique_ptr<byte[]> membufstore = std::make_unique<byte[]>(membufsize);
        memory::buffer membuf = memory::sink(membufstore.get(), membufsize);
        FILE *f = membuf.get();
        
        const int ExtensionIntroducer = 0x21;
        const int ApplicationExtensionLabel = 0xff;
        const int GraphicControlLabel = 0xf9;
        const int CommentLabel = 0xfe;
        const int Trailer = 0x3b;
        
        {
            fputs("GIF89a", f); /// header
        }
        
        {
            /// logical screen descriptor
            const char GlobalColorTableBit = 0x80; /// bit 7
            const char ColorResolutionMask = 0x70; /// bits 6,5,4
            //const char PaletteIsSortedBit = 0x8; /// bit 3
            const char GlobalColorTableSizeMask = 0x7; ///bits 2,1,0
            short width = gif->width;
            short height = gif->height;
            char packed = 0;
            char bgColorIndex = 255;
            char pixelAspectRatio = 0;
            packed |= GlobalColorTableBit;
            const char BitsPerColor = 8;
            packed |= ((BitsPerColor-1) << 4) & ColorResolutionMask;
            packed |= (BitsPerColor-1) & GlobalColorTableSizeMask;
            fwrite(&width, 2, 1, f);
            fwrite(&height, 2, 1, f);
            fputc(packed, f);
            fputc(bgColorIndex, f);
            fputc(pixelAspectRatio, f);
        }
        
        {
            /// global color table
            const int ColorCount = 256;
            if (gif->palette) { fwrite(gif->palette, ColorCount*3, 1, f); }
        }
        
        if (isAnimated(gif)) {
            /// application extension
            fputc(ExtensionIntroducer, f);
            fputc(ApplicationExtensionLabel, f);
            fputc(11, f); /// block size
            fputs("NETSCAPE2.0", f);
            fputc(3, f); /// data block size
            fputc(1, f);
            short repeatCount = 0; /// 0 = loop forever
            fwrite(&repeatCount, 2, 1, f);
            fputc(0, f); /// block terminator
        }
        
        int frameNumber = 0;
        for (Frame *frame = gif->frames, *prevFrame = NULL;
             frame != NULL;
             prevFrame = frame, frame = frame->next, ++frameNumber) {
            MAYBE(FF("frame %d", frameNumber));
            {
                /// graphic control extension
                fputc(ExtensionIntroducer, f);
                fputc(GraphicControlLabel, f);
                fputc(4, f); /// block size
                
                char packed = 0;
                enum {
                    DisposalNotSpecified = 0,
                    DoNotDispose = 1,
                    RestoreToBackgroundColor = 2,
                    RestoreToPrevious = 3
                };
                
                packed |= (DoNotDispose << 2);
                if (isAnimated(gif)) {
                    /// in animated GIFs each frame has transparent pixels
                    /// where it does not differ from previous
                    const int TransparentColorFlag = 1;
                    packed |= TransparentColorFlag;
                }
                
                /// no transparent color index
                fputc(packed, f);
                short delay = frame->delay ? frame->delay : gif->frameDelay;
                fwrite(&delay, 2, 1, f);
                fputc(TranspColorIndex, f); /// transparent color index (if flag is set)
                fputc(0, f); /// block terminator
            }
            
            {
                /// image data
                short fLeft = 0,
                      fTop = 0,
                      fWidth = gif->width,
                      fHeight = gif->height;
                unsigned char * image = frame->indexImage;
                if (prevFrame) {
                    image = new unsigned char[gif->width * gif->height];
                    writeTransparentPixelsWhereNotDifferent(
                        prevFrame->indexImage, frame->indexImage, image,
                        gif->width, gif->height, TranspColorIndex);
                    if (1) {
                        /// crop if borders are transparent
                        int cLeft, cRight, cTop, cBottom;
                        calculatePossibleCrop(
                            gif->width, gif->height, image,
                            TranspColorIndex,
                            cLeft, cRight, cTop, cBottom);
                        
                        if (cLeft <= cRight &&
                            cTop <= cBottom &&
                            cLeft > 0 && cTop > 0 &&
                            cRight < gif->width - 1 &&
                            cBottom < gif->height - 1) {
                                
                                fLeft = cLeft;
                                fTop = cTop;
                                fWidth = cRight + 1 - cLeft;
                                fHeight = cBottom + 1 - cTop;
                                unsigned char *cImage = new unsigned char[fWidth * fHeight];
                                
                                for (int y = 0; y < fHeight; y++) {
                                    unsigned char *srcLine = image + (fTop + y) * gif->width + fLeft;
                                    unsigned char *dstLine = cImage + y * fWidth;
                                    memcpy(dstLine, srcLine, fWidth);
                                }
                                
                                delete[] image;
                                image = cImage;
                                MAYBE(FF(" - cropped to %d%% area",
                                       100 * (fWidth*fHeight) / (gif->width*gif->height)));
                        }
                    } /// end if (1)
                }
                
                {
                    /// image descriptor
                    char packed = 0;
                    fputc(0x2c, f); /// separator
                    fwrite(&fLeft, 2, 1, f);
                    fwrite(&fTop, 2, 1, f);
                    fwrite(&fWidth, 2, 1, f);
                    fwrite(&fHeight, 2, 1, f);
                    fputc(packed, f);
                }
                
                if (1) {
                    const int CodeSize = 8, MaxCodeSize = 12;
                    fputc(CodeSize, f);
                    BlockWriter blockWriter(f);
                    encode(blockWriter, image, fWidth*fHeight, CodeSize, MaxCodeSize);
                    blockWriter.finish();
                    MAYBE(FF(" - %d bytes", blockWriter.totalBytesWritten));
                } else {
                    WTF("Using uncompressed method");
                    int codeSize = 8;
                    int clearCode = 1 << codeSize;
                    int endOfInfo = clearCode + 1;
                    fputc(codeSize, f);
                    BlockWriter blockWriter(f);
                    for (int y = 0; y < gif->height; y++) {
                        for (int x = 0; x < gif->width; x++) {
                            if (x % 100 == 0) { blockWriter.writeBitArray(clearCode, codeSize+1); }
                            /// TODO: a cleverer way to calculate the table reset time...?
                            int c = frame->indexImage[y*gif->width+x];
                            blockWriter.writeBitArray(c, codeSize+1);
                        }
                    }
                    blockWriter.writeBitArray(endOfInfo, codeSize+1);
                    blockWriter.finish();
                }
                if (image != frame->indexImage) { delete[] image; }
            }
            //printf("\n");
        }
        
        {
            /// comment extension
            fputc(ExtensionIntroducer, f);
            fputc(CommentLabel, f);
            const char *CommentText = "(c) Objects In Space And Time LLC";
            const int blockSize = strlen(CommentText);
            if (blockSize <= 255) {
                fputc(blockSize, f);
                fputs(CommentText, f);
            }
            fputc(0, f); /// block terminator
        }
        fputc(Trailer, f);
        
        /// get current position
        std::fflush(f);
        long datasize = std::ftell(f);
        overflow.resize(datasize);
        std::memcpy(&overflow[0], membufstore.get(), datasize);
        
        MAYBE("Done.",
            "About to call overflow.shrink_to_fit()",
         FF("Current byte vector size: %d", overflow.size()));
        
        overflow.shrink_to_fit();
        return overflow;
    }

} /// namespace gif is OVER
