use std::ffi::{c_char, CString};

#[no_mangle]
pub static SHADER_STRING: &[u8] = include_bytes!("../kernel.cl");

#[no_mangle]
pub extern "C" fn rust_function() {
    println!("Hello from Rust!");
    println!(
        "Shader:\n{}",
        String::from_utf8(SHADER_STRING.to_vec()).unwrap()
    );
}

#[no_mangle]
pub extern "C" fn get_shader_source() -> *const c_char {
    let string = String::from_utf8(SHADER_STRING.to_vec()).unwrap();

    let c_string = CString::new(string).unwrap();
    c_string.into_raw()
}
