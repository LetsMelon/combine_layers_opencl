use ocl::{Buffer, MemFlags, ProQue, SpatialDims};

static KERNEL_SRC: &'static str = include_str!("./shader/kernel.cl");

const WIDTH: usize = 4;
const HEIGHT: usize = 4;
const COUNT: usize = 2;

const COLOR_RED: u32 = (0xFF << (8 * 3)) | 0xFF;
const COLOR_GREEN: u32 = (0xFF << (8 * 2)) | 0xFF;
const COLOR_BLUE: u32 = (0xFF << (8 * 1)) | 0xFF;

fn basics() -> ocl::Result<()> {
    // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(SpatialDims::Two(WIDTH, HEIGHT))
        .build()
        .expect("Build ProQue");

    // Create a temporary init vector and the source buffer.
    let vec_source = (0..COUNT)
        .map(|i| match i {
            0 => vec![COLOR_RED; WIDTH * HEIGHT],
            1 => vec![COLOR_GREEN; WIDTH * HEIGHT],
            _ => vec![COLOR_BLUE; WIDTH * HEIGHT],
        })
        .flatten()
        .collect::<Vec<_>>();
    // let vec_source = vec![0xFF0000FF_u32; WIDTH * HEIGHT * COUNT];
    // for i in 0..(WIDTH * HEIGHT) {
    //     for l in 0..COUNT {
    //         let index = i + WIDTH * HEIGHT * l;
    //
    //         let r = i % 0xff;
    //         let g = 0xff;
    //         let b = 0x00;
    //         let a = (l + 0xaa) % 0xff;
    //
    //         vec_source[index] = (r << 24) | (g << 16) | (b << 8) | a;
    //
    //         // 0x00ff00aa
    //         //   r g b a
    //
    //         // vec_source[i + WIDTH * HEIGHT * l] =
    //         //     (0x00FF0000 + ((l + 0xAA) % 0xFF) + ((i % 0xFF) << 24)) as u32;
    //     }
    // }

    vec_source.chunks(WIDTH * HEIGHT).for_each(|chunk| {
        chunk.iter().for_each(|x| print!("0x{:08x}, ", x));
        println!("");
    });

    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(vec_source.len())
        .copy_host_slice(&vec_source)
        .build()?;

    // Create an empty vec and buffer (the quick way) for results. Note that
    // there is no need to initialize the buffer as we did above because we
    // will be writing to the entire buffer first thing, overwriting any junk
    // data that may be there.
    let mut vec_result = vec![0_u32; WIDTH * HEIGHT];
    let result_buffer: Buffer<u32> = ocl_pq.create_buffer()?;

    let kern = ocl_pq
        .kernel_builder("combine_layers")
        .arg(None::<&Buffer<u32>>)
        .arg(None::<&Buffer<u32>>)
        .arg(WIDTH)
        .arg(HEIGHT)
        .arg(COUNT)
        .build()?;

    kern.set_arg(0, Some(&source_buffer))?;
    kern.set_arg(1, &result_buffer)?;
    kern.set_arg(2, &WIDTH)?;
    kern.set_arg(3, &WIDTH)?;
    kern.set_arg(4, &COUNT)?;

    println!(
        "Kernel global work size: {:?}",
        kern.default_global_work_size()
    );

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    result_buffer.read(&mut vec_result).enq()?;
    for idx in 0..(WIDTH * HEIGHT) {
        print!("0x{:08x}, ", vec_result[idx]);
    }
    println!("");

    assert!(
        vec_result[0] == 0x00f600ff,
        "index: 0, 0x{:08x}",
        vec_result[0]
    );
    assert!(
        vec_result[1] == 0x00f600ff,
        "index: 1, 0x{:08x}",
        vec_result[1]
    );
    assert!(
        vec_result[2] == 0x01f600ff,
        "index: 2, 0x{:08x}",
        vec_result[2]
    );
    assert!(
        vec_result[3] == 0x02f600ff,
        "index: 3, 0x{:08x}",
        vec_result[3]
    );
    assert!(
        vec_result[4] == 0x03f600ff,
        "index: 4, 0x{:08x}",
        vec_result[4]
    );
    assert!(
        vec_result[5] == 0x04f600ff,
        "index: 5, 0x{:08x}",
        vec_result[5]
    );
    assert!(
        vec_result[6] == 0x05f600ff,
        "index: 6, 0x{:08x}",
        vec_result[6]
    );
    assert!(
        vec_result[7] == 0x06f600ff,
        "index: 7, 0x{:08x}",
        vec_result[7]
    );
    assert!(
        vec_result[8] == 0x07f600ff,
        "index: 8, 0x{:08x}",
        vec_result[8]
    );
    assert!(
        vec_result[9] == 0x08f600ff,
        "index: 9, 0x{:08x}",
        vec_result[9]
    );
    assert!(
        vec_result[10] == 0x09f600ff,
        "index: 10, 0x{:08x}",
        vec_result[10]
    );
    assert!(
        vec_result[11] == 0x0af600ff,
        "index: 11, 0x{:08x}",
        vec_result[11]
    );
    assert!(
        vec_result[12] == 0x0bf600ff,
        "index: 12, 0x{:08x}",
        vec_result[12]
    );
    assert!(
        vec_result[13] == 0x0cf600ff,
        "index: 13, 0x{:08x}",
        vec_result[13]
    );
    assert!(
        vec_result[14] == 0x0df600ff,
        "index: 14, 0x{:08x}",
        vec_result[14]
    );
    assert!(
        vec_result[15] == 0x0ef600ff,
        "index: 15, 0x{:08x}",
        vec_result[15]
    );

    Ok(())
}

pub fn main() {
    match basics() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}
