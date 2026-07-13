pub mod mul_by_12 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/examples/shaders/mul_by_12.glsl"
    }
}
