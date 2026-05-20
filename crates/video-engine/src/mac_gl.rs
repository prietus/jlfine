//! macOS OpenGL FFI for the `CAOpenGLLayer` mpv host.
//!
//! macOS exposes OpenGL through Apple's CGL API (Core OpenGL) plus
//! the standard `gl*` functions from `OpenGL.framework`. Both are
//! formally deprecated since macOS 10.14 but still ship in 14+ and
//! work for our use case — libmpv's render context understands the
//! "opengl" API just fine on macOS.
//!
//! We only need a handful of CGL calls (pixel-format + context
//! creation) and two GL calls (query current FBO and viewport size)
//! per frame, so we declare them by hand instead of pulling a 1000-
//! function `gl` crate.
//!
//! `CFBundleGetBundleWithIdentifier("com.apple.opengl")` +
//! `CFBundleGetFunctionPointerForName(...)` resolves any GL symbol
//! by name — that's what we hand to libmpv as `get_proc_address`.

use std::ffi::{CStr, c_void};

// ---------------------------------------------------------------- CGL

pub type CGLPixelFormatObj = *mut c_void;
pub type CGLContextObj = *mut c_void;
/// `GLint` from `gltypes.h` is a signed 32-bit int on macOS.
pub type GLint = i32;
pub type GLenum = u32;
pub type GLbitfield = u32;
pub type CGLError = i32;

#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CGLPixelFormatAttribute {
    /// Terminator for the attribute list.
    End = 0,
    Accelerated = 73,
    DoubleBuffer = 5,
    ColorSize = 8,
    ColorFloat = 58,
    OpenGLProfile = 99,
    AllowOfflineRenderers = 96,
    SupportsAutomaticGraphicsSwitching = 101,
}

/// Profile selector for `kCGLPFAOpenGLProfile`. `Core_3_2` requests
/// a modern core profile (no fixed-function pipeline).
#[allow(non_upper_case_globals, dead_code)]
pub const kCGLOGLPVersion_Legacy: u32 = 0x1000;
#[allow(non_upper_case_globals)]
pub const kCGLOGLPVersion_3_2_Core: u32 = 0x3200;
#[allow(non_upper_case_globals, dead_code)]
pub const kCGLOGLPVersion_GL4_Core: u32 = 0x4100;

#[allow(non_upper_case_globals)]
pub const kCGLCPSwapInterval: i32 = 222;
#[allow(non_upper_case_globals)]
pub const kCGLCEMPEngine: i32 = 313;

#[link(name = "OpenGL", kind = "framework")]
unsafe extern "C" {
    pub fn CGLChoosePixelFormat(
        attribs: *const u32,
        pix: *mut CGLPixelFormatObj,
        npix: *mut GLint,
    ) -> CGLError;
    pub fn CGLDestroyPixelFormat(pix: CGLPixelFormatObj) -> CGLError;
    pub fn CGLCreateContext(
        pix: CGLPixelFormatObj,
        share: CGLContextObj,
        ctx: *mut CGLContextObj,
    ) -> CGLError;
    pub fn CGLDestroyContext(ctx: CGLContextObj) -> CGLError;
    pub fn CGLSetCurrentContext(ctx: CGLContextObj) -> CGLError;
    pub fn CGLSetParameter(ctx: CGLContextObj, pname: i32, params: *const GLint) -> CGLError;
    pub fn CGLEnable(ctx: CGLContextObj, pname: i32) -> CGLError;

    pub fn glClear(mask: GLbitfield);
    pub fn glFlush();
    pub fn glGetIntegerv(pname: GLenum, params: *mut GLint);
}

pub const GL_COLOR_BUFFER_BIT: GLbitfield = 0x4000;
pub const GL_DRAW_FRAMEBUFFER_BINDING: GLenum = 0x8CA6;
pub const GL_VIEWPORT: GLenum = 0x0BA2;

// ---------------------------------------------------------------- get_proc_address

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFBundleGetBundleWithIdentifier(bundle_id: *const c_void) -> *const c_void;
    fn CFBundleGetFunctionPointerForName(
        bundle: *const c_void,
        name: *const c_void,
    ) -> *mut c_void;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const i8,
        encoding: u32,
    ) -> *const c_void;
    fn CFRelease(cf: *const c_void);
}

const K_CF_STRING_ENCODING_ASCII: u32 = 0x0600;

/// Look up an OpenGL function pointer by name. Resolved through
/// `OpenGL.framework` (`com.apple.opengl`); hand this to libmpv as
/// the `get_proc_address` callback.
pub fn opengl_get_proc_address(name: &str) -> *mut c_void {
    static BUNDLE_ID: &CStr = c"com.apple.opengl";
    unsafe {
        let bundle_id_cf = CFStringCreateWithCString(
            std::ptr::null(),
            BUNDLE_ID.as_ptr(),
            K_CF_STRING_ENCODING_ASCII,
        );
        if bundle_id_cf.is_null() {
            return std::ptr::null_mut();
        }
        let bundle = CFBundleGetBundleWithIdentifier(bundle_id_cf);
        CFRelease(bundle_id_cf);
        if bundle.is_null() {
            return std::ptr::null_mut();
        }
        let c_name = match std::ffi::CString::new(name) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let name_cf = CFStringCreateWithCString(
            std::ptr::null(),
            c_name.as_ptr(),
            K_CF_STRING_ENCODING_ASCII,
        );
        if name_cf.is_null() {
            return std::ptr::null_mut();
        }
        let func = CFBundleGetFunctionPointerForName(bundle, name_cf);
        CFRelease(name_cf);
        func
    }
}
