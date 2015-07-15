#ifndef __neuquant_inl__
#define __neuquant_inl__

#ifndef __neuquant_h__
#error "neuquant.inl should only be included by neuquant.h"
#endif

// *********************************
// ******* WRAPPER FUNCTIONS *******
// *********************************

inline bool NeuQuant_RGB_to_16(u8 *input, unsigned int width,
                               unsigned int height, u8 *palette,
                               u8 *remapped_image, unsigned int quality) {
    Palette<16, 3> pal;         // our palette
    NeuralQuantizer<16, 3> net; // our neural net

    // add image data to network
    if (!net.AddImage(input, width, height)) {
        return false;
    }

    // quantize
    if (!net.Run(pal, quality)) {
        return false;
    }

    // remap to palette
    pal.ReMap(input, width, height, remapped_image);

    // copy palette data back to user
    pal.CopyToMemory(palette);

    // finished
    return true;
}

inline bool NeuQuant_RGB_to_256(u8 *input, unsigned int width,
                                unsigned int height, u8 *palette,
                                u8 *remapped_image, unsigned int quality) {
    Palette<256, 3> pal;         // our palette
    NeuralQuantizer<256, 3> net; // our neural net

    // add image data to network
    if (!net.AddImage(input, width, height)) {
        return false;
    }

    // quantize
    if (!net.Run(pal, quality)) {
        return false;
    }

    // remap to palette
    pal.ReMap(input, width, height, remapped_image);

    // copy palette data back to user
    pal.CopyToMemory(palette);

    // finished
    return true;
}

inline bool NeuQuant_RGBA_to_16(u8 *input, unsigned int width,
                                unsigned int height, u8 *palette,
                                u8 *remapped_image, unsigned int quality) {
    Palette<16, 4> pal;         // our palette
    NeuralQuantizer<16, 4> net; // our neural net

    // add image data to network
    if (!net.AddImage(input, width, height)) {
        return false;
    }

    // quantize
    if (!net.Run(pal, quality)) {
        return false;
    }

    // remap to palette
    pal.ReMap(input, width, height, remapped_image);

    // copy palette data back to user
    pal.CopyToMemory(palette);

    // finished
    return true;
}

inline bool NeuQuant_RGBA_to_256(u8 *input, unsigned int width,
                                 unsigned int height, u8 *palette,
                                 u8 *remapped_image, unsigned int quality) {
    Palette<256, 4> pal;         // our palette
    NeuralQuantizer<256, 4> net; // our neural net

    // add image data to network
    if (!net.AddImage(input, width, height)) {
        return false;
    }

    // quantize
    if (!net.Run(pal, quality)) {
        return false;
    }

    // remap to palette
    pal.ReMap(input, width, height, remapped_image);

    // copy palette data back to user
    pal.CopyToMemory(palette);

    // finished
    return true;
}

//
// note that the only channel the algorithm cares about is the green channel,
// therefore, RGB is the same as BGR
//

inline bool NeuQuant_BGR_to_16(u8 *input, unsigned int width,
                               unsigned int height, u8 *palette,
                               u8 *remapped_image, unsigned int quality) {
    return NeuQuant_RGB_to_16(input, width, height, palette, remapped_image,
                              quality);
}

inline bool NeuQuant_BGR_to_256(u8 *input, unsigned int width,
                                unsigned int height, u8 *palette,
                                u8 *remapped_image, unsigned int quality) {
    return NeuQuant_RGB_to_256(input, width, height, palette, remapped_image,
                               quality);
}

inline bool NeuQuant_BGRA_to_16(u8 *input, unsigned int width,
                                unsigned int height, u8 *palette,
                                u8 *remapped_image, unsigned int quality) {
    return NeuQuant_RGBA_to_16(input, width, height, palette, remapped_image,
                               quality);
}

inline bool NeuQuant_BGRA_to_256(u8 *input, unsigned int width,
                                 unsigned int height, u8 *palette,
                                 u8 *remapped_image, unsigned int quality) {
    return NeuQuant_RGBA_to_256(input, width, height, palette, remapped_image,
                                quality);
}

// ******************************
// ******* IMPLEMENTATION *******
// ******************************

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Palette() {
    // initialize to zero
    for (unsigned int color = 0; color < NCOLORS; color++) {
        for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
            data[color][channel] = 0;
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Palette(u8 *paldata) {
    // copy from memory
    for (unsigned int color = 0; color < NCOLORS; color++) {
        for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
            // copy element
            data[color][channel] = *paldata;

            // move forward
            paldata++;
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Palette(real *paldata) {
    // copy from real array in memory
    for (unsigned int color = 0; color < NCOLORS; color++) {
        for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
            // convert & copy
            data[color][channel]
                = Clamp(*paldata + real(0.5), real(0), real(255));

            // move forward
            paldata++;
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Palette(
    const Palette &other) {
    if (this == &other) {
        return;
    }

    // copy from other
    for (unsigned int color = 0; color < NCOLORS; color++) {
        for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
            // copy element
            data[color][channel] = other.data[color][channel];
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL> &
    Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::
    operator=(const Palette &rhs) {
    if (this != &rhs) {
        // copy from other
        for (unsigned int color = 0; color < NCOLORS; color++) {
            for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
                // copy element
                data[color][channel] = rhs.data[color][channel];
            }
        }
    }

    return *this;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::~Palette() {
    // empty
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::CopyToMemory(u8 *ptr) {
    // copy to memory
    for (unsigned int color = 0; color < NCOLORS; color++) {
        for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
            // copy element
            *ptr = data[color][channel];

            // move forward
            ptr++;
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void
Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::CopyToMemory(real *ptr) {
    // copy to memory
    for (unsigned int color = 0; color < NCOLORS; color++) {
        for (unsigned int channel = 0; channel < NCHANNELS; channel++) {
            // copy element
            *ptr = real(data[color][channel]);

            // move forward
            ptr++;
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::NumChannels() {
    return NCHANNELS;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::NumColors() {
    return NCOLORS;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::NumElements() {
    return NumChannels() * NumColors();
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int
Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::GreenChannelID() {
    return GREEN_CHANNEL;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Sort() {
    // simple bubble sort (should be OK, size never likely to exceed 256)
    for (unsigned int i = NCOLORS; i > 0; i--) {
        for (unsigned int j = 1; j < i; j++) {
            if (Compare(j - 1, j) > 0) {
                Swap(j - 1, j);
            }
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline bool Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::IsSorted() {
    for (unsigned int i = 1; i < NCOLORS; i++) {
        if (Compare(i - 1, i) > 0) {
            return false;
        }
    }

    return true;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::ReMap(
    u8 *image, unsigned int width, unsigned int height, u8 *newimage) {
    unsigned int pixels = width * height;
    bool sorted = IsSorted();

    if (IsSorted()) {
        // store the smallest distance we've found in a match (this becomes our
        // search neighborhood)
        unsigned int d_min = NCOLORS;

        for (unsigned int i = 0; i < pixels; i++) {
            unsigned int center = FindClosestGreen(&(image[i * NCHANNELS]));

            // find min/max search range
            int minsearch
                = Clamp<int>((int)center - (int)d_min, 0, NCOLORS - 1);
            int maxsearch
                = Clamp<int>((int)center + (int)d_min, 0, NCOLORS - 1);

            // perform search
            newimage[i]
                = FindClosest(&(image[i * NCHANNELS]), minsearch, maxsearch);

            // store distance
            unsigned int newdist
                = PerceptualDistance(&(image[i * NCHANNELS]), newimage[i]);
            if (newdist < d_min) {
                d_min = Clamp(newdist, NCOLORS / 4, NCOLORS);
            }
        }
    } else {
        // brute force
        for (unsigned int i = 0; i < pixels; i++) {
            // search everything, store pixel
            newimage[i] = FindClosest(&(image[i * NCHANNELS]), 0, NCOLORS - 1);
        }
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int
Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::FindClosestGreen(u8 *color) {
    // binary search
    int low = 0;
    int high = NCOLORS - 1;
    int middle;

    while (low <= high) {
        // find middle for our comparison
        middle = (low + high) / 2;

        // look for a match or which direction to continue search
        if (color[GREEN_CHANNEL] == data[middle][GREEN_CHANNEL]) {
            // matched
            return (unsigned int)middle;
        } else if (color[GREEN_CHANNEL] < data[middle][GREEN_CHANNEL]) {
            // search lower
            high = middle - 1;
        } else {
            // search higher
            low = middle + 1;
        }
    }

    // not found, return closest
    return (unsigned int)Clamp<int>(low, 0, NCOLORS - 1);
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::FindClosest(
    u8 *color, unsigned int minentry, unsigned int maxentry) {
    // store our best match
    unsigned int mindist = 1000000000;
    unsigned int minindex = 0;

    // search all colors for match
    for (unsigned int i = minentry; i <= maxentry; i++) {
        unsigned int dist = PerceptualDistance(color, i);
        if (dist < mindist) {
            mindist = dist;
            minindex = i;
        }
    }

    // return best match
    return minindex;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int
Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::PerceptualDistance(
    u8 *color, unsigned int entry) {
    unsigned int d = 0;
    for (unsigned int i = 0; i < NCHANNELS; i++) {
        d += Abs((int)color[i] - (int)data[entry][i]);
    }
    return d;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline int Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Compare(unsigned int a,
                                                               unsigned int b) {
    if (data[a][GREEN_CHANNEL] < data[b][GREEN_CHANNEL]) {
        return -1;
    } else if (data[a][GREEN_CHANNEL] > data[b][GREEN_CHANNEL]) {
        return 1;
    } else {
        // green channels equal, test all others
        for (unsigned int i = 0; i < NCHANNELS; i++) {
            if (i != GREEN_CHANNEL) {
                if (data[a][i] < data[b][i]) {
                    return -1;
                } else if (data[a][i] > data[b][i]) {
                    return 1;
                }
            }
        }
    }

    // exactly equal
    return 0;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Swap(unsigned int a,
                                                             unsigned int b) {
    for (unsigned int i = 0; i < NCHANNELS; i++) {
        u8 tmp = data[a][i];
        data[a][i] = data[b][i];
        data[b][i] = tmp;
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::NeuralQuantizer()
    : image(0), image_size(0) {
    // initialize network
    for (unsigned int i = 0; i < NCOLORS; i++) {
        // initial weights [0,1]
        neurons[i].SetAllChannels((real)i / (real)(NCOLORS - 1) * real(255));
    }

    // initial gamma, beta
    gamma = real(4) * real(NCOLORS);
    beta = real(1) / gamma;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::~NeuralQuantizer() {
    delete[] image;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline bool NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::AddImage(
    u8 *input, unsigned int width, unsigned int height) {
    // remember image sizes
    unsigned int this_image_size = width * height * NCHANNELS;
    unsigned int prev_image_size = image_size;
    unsigned int new_image_size = prev_image_size + this_image_size;

    // allocate
    real *tmpimage = new real[new_image_size];
    if (!tmpimage) {
        return false;
    }

    // convert original image bytes to reals
    for (unsigned int i = 0; i < this_image_size; i++) {
        tmpimage[i] = real(input[i]);
    }

    // now copy any prior images over
    for (unsigned int i = this_image_size; i < new_image_size; i++) {
        tmpimage[i] = image[i - this_image_size];
    }

    // delete old image array
    delete[] image;

    // swap in new image array
    image = tmpimage;
    image_size = new_image_size;

    return true;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline bool NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Run(
    Palette<NCOLORS, NCHANNELS, GREEN_CHANNEL> &palette, unsigned int quality) {
    // make sure we have an image
    if (!image || !image_size) {
        return false;
    }

    // clamp quality
    quality = Clamp<unsigned int>(quality, 1, 10);

    // train network
    unsigned int total_pels = (image_size / NCHANNELS);
    unsigned int pels_to_sample = total_pels / quality;
    unsigned int pels_per_cycle = pels_to_sample / 100;
    unsigned int next_pel = 0;
    unsigned int prime = ChoosePrime();
    unsigned int total_cycles = 101;

    for (unsigned int cycle = 0; cycle < total_cycles; cycle++) {
        // if this is our last cycle let's do the remainder so we use all pels
        // at least once
        if (cycle == 100) {
            pels_per_cycle = pels_to_sample % 100;
        }

        // calculate alpha
        real alpha = exp(real(-0.03) * real(cycle));

        // calculate radius
        unsigned int radius = (unsigned int)(floor(
            real(NCOLORS) / real(8) * exp(real(-0.0325) * real(cycle))));

        // present image data
        for (unsigned int i = 0; i < pels_per_cycle; i++) {
            next_pel += prime;
            if (next_pel >= total_pels) {
                next_pel -= total_pels;
            }
            Present(&(image[next_pel * NCHANNELS]), alpha, radius);
        }
    }

    // write palette
    for (unsigned int i = 0; i < NCOLORS; i++) {
        for (unsigned int j = 0; j < NCHANNELS; j++) {
            palette.data[i][j] = u8(
                Clamp(neurons[i].weights[j] + real(0.5), real(0), real(255)));
        }
    }

    // sort palette
    palette.Sort();

    return true;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Neuron::Neuron() {
    bias = real(0);
    frequency = real(1) / real(NCOLORS);
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void
NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Neuron::SetAllChannels(
    real value) {
    for (unsigned int i = 0; i < NCHANNELS; i++) {
        weights[i] = value;
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline real
NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Neuron::DesienoDistance(
    real *color) const {
    real dist = real(0);
    for (unsigned int i = 0; i < NCHANNELS; i++) {
        dist += Abs(color[i] - weights[i]);
    }
    dist -= bias;

    return dist;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Neuron::
    UpdateLoserFrequencyAndBias(real *color, real beta, real gamma) {
    // update frequency of non winner
    frequency = frequency - beta * frequency;

    // update bias of non winner
    bias = bias + gamma * beta * frequency;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Neuron::
    UpdateWinnerFrequencyAndBias(real *color, real beta, real gamma) {
    // update frequency of winner
    frequency = frequency - beta * frequency + beta;

    // update bias of winner
    bias = bias + gamma * beta * frequency - gamma * beta;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void
NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Neuron::UpdateWeight(
    real *color, real alpha, real rho) {
    for (unsigned int i = 0; i < NCHANNELS; i++) {
        // update weight
        weights[i]
            = (alpha * rho * color[i]) + (real(1) - (alpha * rho)) * weights[i];
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int
NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::FindWinningNeuron(
    real *color) {
    // find winning neuron
    real windist = real(1000000000000);
    unsigned int winindex = NCOLORS / 2;

    for (unsigned int i = 0; i < NCOLORS; i++) {
        // calculate distance
        real dist = neurons[i].DesienoDistance(color);

        // record new best distance
        if (dist < windist) {
            windist = dist;
            winindex = i;
        }
    }

    return winindex;
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline void NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Present(
    real *color, real alpha, unsigned int radius) {
    // search for winning neuron
    int winindex = (int)FindWinningNeuron(color);

    // update frequency & bias
    for (unsigned int i = 0; i < NCOLORS; i++) {
        if (i == winindex) {
            neurons[i].UpdateWinnerFrequencyAndBias(color, beta, gamma);
        } else {
            neurons[i].UpdateLoserFrequencyAndBias(color, beta, gamma);
        }
    }

    // determine neighbors
    int min_neighbor = Clamp<int>(winindex - (int)radius, 0, NCOLORS - 1);
    int max_neighbor = Clamp<int>(winindex + (int)radius, 0, NCOLORS - 1);

    // update neighbors
    for (int i = min_neighbor; i <= max_neighbor; i++) {
        // update neuron
        neurons[i].UpdateWeight(color, alpha, Rho(i, winindex, radius));
    }
}

template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline real
NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::Rho(int i, int j,
                                                        unsigned int radius) {
    if (i == j) {
        return real(1);
    } else {
        real tmp = real(Abs(j - i)) / real(radius);
        return real(1) - (tmp * tmp);
    }
}

// choose prime near 500 which isn't a multiple of width*height
template <unsigned int NCOLORS, unsigned int NCHANNELS,
          unsigned int GREEN_CHANNEL>
inline unsigned int
NeuralQuantizer<NCOLORS, NCHANNELS, GREEN_CHANNEL>::ChoosePrime() {
    if (image_size % 487) {
        return 487;
    } else if (image_size % 491) {
        return 491;
    } else if (image_size % 499) {
        return 499;
    } else if (image_size % 503) {
        return 503;
    } else {
        return 487; // TODO: assert? ever possible? or nearly completely
                    // unlikely??? prime stuff...
    }
}

#endif // __neuquant_inl__
