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
