use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Debug, Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
pub struct Vertex2Df {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl Vertex2Df {
    pub const fn new_xy(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }

    pub const fn x(&self) -> f32 {
        self.position[0]
    }

    pub const fn y(&self) -> f32 {
        self.position[1]
    }
}

pub struct Shape2D {
    pub vertices: Vec<Vertex2Df>,
}

impl Shape2D {
    pub fn new(vertices: impl Into<Vec<Vertex2Df>>) -> Self {
        Self {
            vertices: vertices.into(),
        }
    }
}
