pub mod app;
pub mod error;
pub(crate) mod examples;
pub mod shape;
pub mod vk;

use clap::Parser;
use tracing::Level;

#[derive(Parser, Debug)]
#[clap(author = "Zachary Cauchi", version, about)]
/// Application configuration
struct Args {
    /// Max log-level
    #[arg(short, long, default_value_t = Level::INFO)]
    level: Level,

    /// an optional name to greet
    #[arg()]
    name: Option<String>,
}

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt().with_max_level(args.level).init();

    println!("App started.",);

    if let Err(e) = app::run() {
        tracing::error!("Error occurred during app runtime. Error: {e}");
    }
}
