
// https://doc.rust-lang.org/stable/cargo/reference/build-scripts.html
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");    // XXX: prevent re-run indead
    // By default, cargo always re-run the build script if any file within the package
    // is changed, and no any rerun-if instruction is emitted.
    //println!("cargo:rerun-if-changed=src");
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}",
        chrono::Local::now().format("%H:%M:%S%z %Y-%m-%d"));

    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"]).output()?;
    println!("cargo:rustc-env=BUILD_GIT_HASH={}", String::from_utf8(output.stdout)?);
    println!("cargo:rerun-if-changed={}", std::path::Path::new(".git").join("index").display());

    let (name, url) = ("mnist.onnx",
        "https://github.com/tracel-ai/burn/raw/main/examples/onnx-inference/src/model");
    //let (name, url) = ("mnist-12-int8.onnx",  // XXX: yet not support by onnx parser?
    //    "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model");

    let onnx = std::path::PathBuf::from("target").join("model");    //env!("OUT_DIR")?)
    std::fs::create_dir_all(&onnx)?;    let onnx = onnx.join(name);
    std::io::Write::write_all(&mut std::fs::File::create(&onnx)?, &burn_common::network::
        downloader::download_file_as_bytes(&format!("{url}/{name}"), name))?;

    burn_import::onnx::ModelGen::new().input(onnx.to_str().unwrap())
        //.record_type(burn_import::onnx::RecordType::Bincode).embed_states(true)
        .out_dir("model").run_from_script();   // bundled/embedded model into the binary

    Ok(())
}

