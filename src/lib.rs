use std::ffi::{c_char, CString};

static SHADER_STRING: &'static str = include_str!("./shader/kernel.cl");

#[no_mangle]
pub extern "C" fn get_shader_source() -> *const c_char {
    let string = SHADER_STRING.to_string();

    match CString::new(string) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => std::ptr::null(),
    }
}
