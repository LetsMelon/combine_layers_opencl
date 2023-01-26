#define COLOR_CHANEL_RED 3
#define COLOR_CHANEL_GREEN 2
#define COLOR_CHANEL_BLUE 1
#define COLOR_CHANEL_ALPHA 0

uint pack_color(uchar4 c) {
  return (c[COLOR_CHANEL_RED] << (COLOR_CHANEL_RED * 8) |
          c[COLOR_CHANEL_GREEN] << (COLOR_CHANEL_GREEN * 8) |
          c[COLOR_CHANEL_BLUE] << (COLOR_CHANEL_BLUE * 8) |
          c[COLOR_CHANEL_ALPHA] << (COLOR_CHANEL_ALPHA * 8));
}

// print_uchar4s(uchar4 c1, uchar4 c2) {
//   printf("c1: r: %02x, g: %02x, b: %02x, a: %02x\t\tc2: r: %02x, g: %02x, b:
//   "
//          "%02x, a: "
//          "%02x\n",
//          c1[COLOR_CHANEL_RED], c1[COLOR_CHANEL_GREEN], c1[COLOR_CHANEL_BLUE],
//          c1[COLOR_CHANEL_ALPHA], c2[COLOR_CHANEL_RED],
//          c2[COLOR_CHANEL_GREEN], c2[COLOR_CHANEL_BLUE],
//          c2[COLOR_CHANEL_ALPHA]);
// }

/*
uchar4 blend(uchar4 fg, uchar4 bg) {
  uchar c_a = fg[COLOR_CHANEL_ALPHA] +
              (255 - fg[COLOR_CHANEL_ALPHA]) * bg[COLOR_CHANEL_ALPHA];
  uchar fract = 255 / c_a;

  uchar c_r = fract * (fg[COLOR_CHANEL_ALPHA] * fg[COLOR_CHANEL_RED] +
                       (255 - fg[COLOR_CHANEL_ALPHA]) * bg[COLOR_CHANEL_ALPHA] *
                           bg[COLOR_CHANEL_RED]);
  uchar c_g = fract * (fg[COLOR_CHANEL_ALPHA] * fg[COLOR_CHANEL_GREEN] +
                       (255 - fg[COLOR_CHANEL_ALPHA]) * bg[COLOR_CHANEL_ALPHA] *
                           bg[COLOR_CHANEL_GREEN]);
  uchar c_b = fract * (fg[COLOR_CHANEL_ALPHA] * fg[COLOR_CHANEL_BLUE] +
                       (255 - fg[COLOR_CHANEL_ALPHA]) * bg[COLOR_CHANEL_ALPHA] *
                           bg[COLOR_CHANEL_BLUE]);
  return (uchar4)(c_a, c_b, c_g, c_r);
}
*/

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

    uchar4 result = as_uchar4(input_buffer[index]);

    for (uint i = 1; i < count; i += 1) {
      uint layer_index = index + width * height * i;

      uchar4 fg = as_uchar4(input_buffer[layer_index]);
      result = blend(fg, result);
    }

    output_buffer[index] = pack_color(result);
  }
}
