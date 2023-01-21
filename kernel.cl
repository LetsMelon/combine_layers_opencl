__kernel void vadd(__global float *a, __global float *b, __global float *c,
                   const unsigned int count) {

  uint i = (uint)get_global_id(0);

  if (i < count) {
    c[i] = a[i] + b[i];
  }
}

#define CHANNEL unsigned char

uint pack_color(CHANNEL r, CHANNEL g, CHANNEL b, CHANNEL a) {
  return (r << 24 | g << 16 | b << 8 | a);
}

void blend(unsigned char result[4], unsigned char fg[4], unsigned char bg[4]) {
  unsigned int alpha = fg[3] + 1;
  unsigned int inv_alpha = 256 - fg[3];
  result[0] = (unsigned char)((alpha * fg[0] + inv_alpha * bg[0]) >> 8);
  result[1] = (unsigned char)((alpha * fg[1] + inv_alpha * bg[1]) >> 8);
  result[2] = (unsigned char)((alpha * fg[2] + inv_alpha * bg[2]) >> 8);
  result[3] = 0xff;
}

__kernel void combine_layers(__global unsigned int *input_buffer,
                             __global unsigned int *output_buffer,
                             const unsigned int width,
                             const unsigned int height,
                             const unsigned int count) {
  uint x = (uint)get_global_id(0);
  uint y = (uint)get_global_id(1);

  if ((x < width) && (y < height) && (count > 0)) {
    uint index = x + y * height;

    unsigned char result[4];

    for (uint ii = 0; ii < count; ii += 1) {
      uint layer_index = index + width * height * ii;

      uint c = input_buffer[layer_index];
      unsigned char fg[4] = {
          (unsigned char)(c >> 24 & 0xFF), (unsigned char)(c >> 16 & 0xFF),
          (unsigned char)(c >> 8 & 0xFF), (unsigned char)(c >> 0 & 0xFF)};
      blend(result, fg, result);
    }

    output_buffer[index] =
        pack_color(result[0], result[1], result[2], result[3]);
  }
}
