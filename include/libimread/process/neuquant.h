/*
 *  Copyright (c) 2010, Stephen Waits <steve@waits.net>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to
 * deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#ifndef __neuquant_h__
#define __neuquant_h__

/*
 * An implementation of the "neuquant" color quantization algorithm in
 * C++. Primarily inlined and templated.
 *
 */

/**
 * 'real' represents the floating point format used throughout this
 * NeuQuant implementation.
 */
typedef float real;

/**
 * 'u8' is used mostly for shorthand, anywhere a byte or pointer
 * to a byte is needed
 */
typedef unsigned char u8;

/**
 * Wrapper function to run NeuQuant and remap an RGB image to a 16 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_RGB_to_16(u8 *input, unsigned int width, unsigned int height,
                        u8 *palette, u8 *remapped_image,
                        unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap an RGB image to a 256 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_RGB_to_256(u8 *input, unsigned int width, unsigned int height,
                         u8 *palette, u8 *remapped_image,
                         unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap an RGBA image to a 16 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_RGBA_to_16(u8 *input, unsigned int width, unsigned int height,
                         u8 *palette, u8 *remapped_image,
                         unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap an RGBA image to a 256 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_RGBA_to_256(u8 *input, unsigned int width, unsigned int height,
                          u8 *palette, u8 *remapped_image,
                          unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap a BGR image to a 16 color palette.
 * Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_BGR_to_16(u8 *input, unsigned int width, unsigned int height,
                        u8 *palette, u8 *remapped_image,
                        unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap a BGR image to a 256 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_BGR_to_256(u8 *input, unsigned int width, unsigned int height,
                         u8 *palette, u8 *remapped_image,
                         unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap a BGRA image to a 16 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_BGRA_to_16(u8 *input, unsigned int width, unsigned int height,
                         u8 *palette, u8 *remapped_image,
                         unsigned int quality = 1);

/**
 * Wrapper function to run NeuQuant and remap a BGR image to a 256 color
 * palette.  Uses
 * Palette and NeuralQuantizer classes.
 *
 * For an example of how to use the objects, see the source for this function.
 *
 * @param input   pointer to input image data
 * @param width   width of input image
 * @param height  height of input image
 * @param palette pointer to palette destination
 * @param remapped_image
 *                pointer to remapped image destination
 * @param quality integer in range [1,10], where 1=best, and 10=fastest
 *
 * @return true on success; false otherwise
 */
bool NeuQuant_BGRA_to_256(u8 *input, unsigned int width, unsigned int height,
                          u8 *palette, u8 *remapped_image,
                          unsigned int quality = 1);

/**
 * Absolute value utility function.
 *
 * @param a      value to take absolute of
 *
 * @return absolute value of a
 */
template <typename T>
inline T Abs(T a) {
    if (a < T(0)) {
        return -a;
    } else {
        return a;
    }
}

/**
 * Range clamping utility function.
 *
 * @param a       value to clamp
 * @param minimum minimum value of range (inclusive)
 * @param maximum maximum value of range (inclusive)
 *
 * @return a clamped to [minimum,maximum]
 */
template <typename T>
inline T Clamp(T a, T minimum, T maximum) {
    if (a < minimum) {
        return minimum;
    } else if (a > maximum) {
        return maximum;
    } else {
        return a;
    }
}

/**
 * A color palette.
 */
template <unsigned int NCOLORS, unsigned int NCHANNELS = 3,
          unsigned int GREEN_CHANNEL = 1>
class Palette {
  public:
    /**
     * Palette data
     */
    u8 data[NCOLORS][NCHANNELS];

    /**
     * Default constructer.  Initializes palette to all 0's.
     */
    Palette();
    /**
     * Construct a palette from raw byte data.
     *
     * @param paldata pointer to data
     */
    Palette(u8 *paldata);
    /**
     * Construct a palette from floating point array.
     *
     * @param paldata pointer to floating point data
     */
    Palette(real *paldata);
    /**
     * Copy constructor.
     *
     * @param other
     */
    Palette(const Palette &other);

    /**
     * Copy assignment operator.
     *
     * @param rhs    object to copy
     *
     * @return self reference
     */
    Palette &operator=(const Palette &rhs);

    /**
     * Destructor.
     */
    virtual ~Palette();

    /**
     * Utility function to copy internal palette data to memory.
     *
     * @param ptr    pointer to destination memory
     */
    void CopyToMemory(u8 *ptr);

    /**
     * Utility function to convert internal palette data to floating point
     * and copy to memory.
     *
     * @param ptr    pointer to destination memory
     */
    void CopyToMemory(real *ptr);

    /**
     * Accessor to retreive number of channels in a palette.
     *
     * @return number of color channels in palette.
     */
    unsigned int NumChannels();

    /**
     * Accessor to retrieve palette size.
     *
     * @return size of palette
     */
    unsigned int NumColors();

    /**
     * Accessor to determine total number of elements in palette.
     *
     * @return NumColors() * NumChannels()
     */
    unsigned int NumElements();

    /**
     * Accessor to retrieve the index of the green channel in the palette.
     *
     * @return index of green channel
     */
    unsigned int GreenChannelID();

    /**
     * Sort a palette by its green channel.
     */
    void Sort();

    /**
     * Determine if a palette is sorted (according to Sort() method)
     *
     * @return true if palette is sorted; false otherwise
     */
    bool IsSorted();

    /**
     * Remap 8 bit image data to a palette.
     *
     * @param image    pointer to original full color image data
     * @param width    width of image in pixels
     * @param height   height of image in pixels
     * @param newimage pointer to destination image (should be width*height in
     * size)
     */
    void ReMap(u8 *image, unsigned int width, unsigned int height,
               u8 *newimage);

  private:
    unsigned int PerceptualDistance(u8 *color, unsigned int entry);
    unsigned int FindClosestGreen(u8 *color);
    unsigned int FindClosest(u8 *color, unsigned int minentry,
                             unsigned int maxentry);

    int Compare(unsigned int a, unsigned int b);
    void Swap(unsigned int a, unsigned int b);
};

/**
 * Implementation of NEUQUANT algorithm as described in the paper "Kohonen
 * Neural Networks for Optimal Colour Quantization", by Anthony H. Dekker
 * dekker@ACM.org
 */
template <unsigned int NCOLORS, unsigned int NCHANNELS = 3,
          unsigned int GREEN_CHANNEL = 1>
class NeuralQuantizer {
  public:
    /**
     * Default constructor.  Initializes algorithm settings.
     */
    NeuralQuantizer();

    /**
     * Destructor.  Releases any allocated resources.
     */
    virtual ~NeuralQuantizer();

    /**
     * Add image data to be quantized.  This may be called multiple times to
     * quantize more than one image to a single palette.
     *
     * @param input  pointer to image data
     * @param width  width of image
     * @param height height of image
     *
     * @return true on success; false otherwise
     */
    bool AddImage(u8 *input, unsigned int width, unsigned int height);

    /**
     * Perform the quantization.  Should be called after AddImage() is called
     * at least one time successfully.
     *
     * @param palette reference to output palette - this is where the result
     * goes
     * @param quality an integer in the range [1,10], with 1 == best, and 10 ==
     * fastest
     *
     * @return true on success; false otherwise
     */
    bool Run(Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL> &palette,
             unsigned int quality = 1);

  private:
    struct Neuron {
        real weights[NCHANNELS];
        real bias;
        real frequency;

        Neuron();

        void SetAllChannels(real value);

        real DesienoDistance(real *color) const;

        void UpdateLoserFrequencyAndBias(real *color, real beta, real gamma);
        void UpdateWinnerFrequencyAndBias(real *color, real beta, real gamma);
        void UpdateWeight(real *color, real alpha, real rho);
    };

    // data
    Neuron neurons[NCOLORS];
    real beta;
    real gamma;

    real *image;
    unsigned int image_size;

    unsigned int FindWinningNeuron(real *color);

    void Present(real *color, real alpha, unsigned int radius);

    real Rho(int i, int j, unsigned int radius);

    unsigned int ChoosePrime();
};

#include "neuquant.inl"

#endif // __neuquant_h__
