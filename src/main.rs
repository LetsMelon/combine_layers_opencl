use ocl::{Buffer, MemFlags, ProQue, SpatialDims};
use rusvid_core::plane::*;

static KERNEL_SRC: &'static str = include_str!("./shader/kernel.cl");

const SIZE: usize = 2_usize.pow(10);
const WIDTH: usize = SIZE;
const HEIGHT: usize = SIZE;
const COUNT: usize = 2;

const COLOR_RED: u32 = (0xFF << (8 * 3)) | 0xFF;
const COLOR_GREEN: u32 = (0xFF << (8 * 2)) | 0x77;
const COLOR_BLUE: u32 = (0xFF << (8 * 1)) | 0xFF;

fn basics() -> ocl::Result<()> {
    // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(SpatialDims::Two(WIDTH, HEIGHT))
        .build()
        .expect("Build ProQue");

    // Create a temporary init vector and the source buffer.

    let frames = (0..COUNT)
        .map(|i| match i {
            0 => [
                vec![COLOR_RED; WIDTH * HEIGHT / 2],
                vec![0; WIDTH * HEIGHT / 2],
            ]
            .concat(),
            1 => [
                vec![0; WIDTH * HEIGHT / 4],
                vec![COLOR_GREEN; WIDTH * HEIGHT / 2],
                vec![0; WIDTH * HEIGHT / 4],
            ]
            .concat(),
            _ => vec![COLOR_BLUE; WIDTH * HEIGHT],
        })
        .collect::<Vec<_>>();

    frames.iter().enumerate().for_each(|(i, frame)| {
        let plane = Plane::from_data(
            WIDTH as u32,
            HEIGHT as u32,
            frame
                .iter()
                .map(|pixel| {
                    let r = ((pixel >> (8 * 3)) & 0xFF) as u8;
                    let g = ((pixel >> (8 * 2)) & 0xFF) as u8;
                    let b = ((pixel >> (8 * 1)) & 0xFF) as u8;
                    let a = ((pixel >> (8 * 0)) & 0xFF) as u8;

                    [r, g, b, a]
                })
                .collect(),
        )
        .unwrap();

        plane.save_as_png(format!("frame_{}.png", i)).unwrap();
    });

    let vec_source = frames.iter().cloned().flatten().collect::<Vec<_>>();

    // vec_source.chunks(WIDTH * HEIGHT).for_each(|chunk| {
    //     chunk.iter().for_each(|x| print!("0x{:08x}, ", x));
    //     println!("");
    // });

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

    result_buffer.read(&mut vec_result).enq()?;

    print!("0x{:08x}, ", vec_result[0]);

    let plane = Plane::from_data(
        WIDTH as u32,
        HEIGHT as u32,
        vec_result
            .iter()
            .map(|pixel| {
                let r = ((pixel >> (8 * 3)) & 0xFF) as u8;
                let g = ((pixel >> (8 * 2)) & 0xFF) as u8;
                let b = ((pixel >> (8 * 1)) & 0xFF) as u8;
                let a = ((pixel >> (8 * 0)) & 0xFF) as u8;

                [r, g, b, a]
            })
            .collect(),
    )
    .unwrap();

    plane.save_as_png(format!("output.png")).unwrap();

    /*
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
     */

    Ok(())
}

pub fn main() {
    match basics() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}
