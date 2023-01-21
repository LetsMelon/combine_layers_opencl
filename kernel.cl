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

__kernel void combine_layers(__global unsigned int *input_buffer,
                             __global unsigned int *output_buffer,
                             const unsigned int width,
                             const unsigned int height,
                             const unsigned int count) {
  uint x = (uint)get_global_id(0);
  uint y = (uint)get_global_id(1);

  if ((x < width) && (y < height) && (count > 0)) {
    uint index = x + y * height;

    uint r = 0;
    uint g = 0;
    uint b = 0;
    uint a = 0;

    for (uint ii = 0; ii < count; ii += 1) {
      uint layer_index = index + width * height * ii;

      uint c = input_buffer[layer_index];
      uint c_r = c >> 24 & 0xFF;
      uint c_g = c >> 16 & 0xFF;
      uint c_b = c >> 8 & 0xFF;
      uint c_a = c >> 0 & 0xFF;

      // TODO implement prober algorithm to combine colors with alpha values
      r += c_r;
      g += c_g;
      b += c_b;
      a += c_a;
    }

    float weight = 1.0 / (float)count;
    r = (uint)((float)r * weight);
    g = (uint)((float)g * weight);
    b = (uint)((float)b * weight);
    a = (uint)((float)a * weight);

    output_buffer[index] = pack_color(r, g, b, a);
  }
}
