// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
// ... Only not really (release-ready attribution TBD)
// License: MIT (see COPYING.MIT file)

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <libimread/process/neuquant.hh>

/*
#define FILENAME "t.raw"
#define WIDTH    (510)
#define HEIGHT   (383)
/*/
#define FILENAME "shorts.raw"
#define WIDTH (128)
#define HEIGHT (64)
//*/

int neuquant_main() {
    printf("Loading...\n");
    FILE *f;
    f = fopen(FILENAME, "rb");
    neuquant::u8 texture[WIDTH * HEIGHT * 3];
    fread(texture, WIDTH * HEIGHT * 3, 1, f);
    fclose(f);
    
    printf("Quantizing, Remapping...\n");
    neuquant::u8 newpal[16][3];
    neuquant::u8 newimage[WIDTH * HEIGHT];
    neuquant::NeuQuant_RGB_to_16((neuquant::u8 *)texture, WIDTH, HEIGHT,
                                 (neuquant::u8 *)newpal,
                                 (neuquant::u8 *)newimage, 3);
    
    // unpalettize original (hack so we can write out another RAW)
    for (unsigned int i = 0; i < (WIDTH * HEIGHT); i++) {
        texture[i * 3 + 0] = newpal[newimage[i]][0];
        texture[i * 3 + 1] = newpal[newimage[i]][1];
        texture[i * 3 + 2] = newpal[newimage[i]][2];
    }
    
    // write output (temp, test)
    printf("Saving...\n");
    f = fopen("out" FILENAME, "wb");
    fwrite(texture, WIDTH * HEIGHT * 3, 1, f);
    fclose(f);
    
    return 0;
}

#undef FILENAME
#undef WIDTH
#undef HEIGHT
