pub mod mul_by_12 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/examples/shaders/mul_by_12.glsl"
    }
}

pub mod mandelbrot {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/examples/shaders/mandelbrot.glsl"
    }
}

pub mod vertex_basic {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/examples/shaders/vertex_basic.glsl"
    }
}

pub mod fragment_basic {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/examples/shaders/fragment_basic.glsl"
    }
}
