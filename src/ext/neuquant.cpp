
#include <cmath>
#include <libimread/ext/neuquant.h>

/*
#define FILENAME "t.raw"
#define WIDTH    (510)
#define HEIGHT   (383)
/*/
#define FILENAME "shorts.raw"
#define WIDTH    (128)
#define HEIGHT   (64)
//*/

#include <cstdio>
#include <cstdlib>

int main()
{
    printf("Loading...\n");
    FILE* f;
    f = fopen(FILENAME,"rb");
    u8 texture[WIDTH*HEIGHT*3];
    fread(texture,WIDTH*HEIGHT*3,1,f);
    fclose(f);

    printf("Quantizing, Remapping...\n");
    u8 newpal[16][3];
    u8 newimage[WIDTH*HEIGHT];
    NeuQuant_RGB_to_16((u8*)texture,WIDTH,HEIGHT,(u8*)newpal,(u8*)newimage,3);

    // unpalettize original (hack so we can write out another RAW)
    for (unsigned int i=0;i<(WIDTH*HEIGHT);i++)
    {
        texture[i*3+0] = newpal[newimage[i]][0];
        texture[i*3+1] = newpal[newimage[i]][1];
        texture[i*3+2] = newpal[newimage[i]][2];
    }

    // write output (temp, test)
    printf("Saving...\n");
    f = fopen("out" FILENAME,"wb");
    fwrite(texture,WIDTH*HEIGHT*3,1,f);
    fclose(f);

    return 0;
}



