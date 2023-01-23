__kernel void vadd(__global float *a, __global float *b, __global float *c,
                   const unsigned int count) {

  uint i = (uint)get_global_id(0);

  if (i < count) {
    c[i] = a[i] + b[i];
  }
}

uint pack_color(uchar4 c) {
  return (c[3] << 24 | c[2] << 16 | c[1] << 8 | c[0]);
}

uchar4 blend(uchar4 fg, uchar4 bg) {
  uint alpha = fg[0] + 1;
  uint inv_alpha = 256 - fg[0];

  return (uchar4)(0xff, (uchar)((alpha * fg[1] + inv_alpha * bg[1]) >> 8),
                  (uchar)((alpha * fg[2] + inv_alpha * bg[2]) >> 8),
                  (uchar)((alpha * fg[3] + inv_alpha * bg[3]) >> 8));
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

    uchar4 result;

    for (uint i = 0; i < count; i += 1) {
      uint layer_index = index + width * height * i;

      uchar4 fg = as_uchar4(input_buffer[layer_index]);
      result = blend(fg, result);
    }

    output_buffer[index] = pack_color(result);
  }
}
