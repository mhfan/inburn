
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{tensor::Tensor, backend::Wgpu as Backend};
    //type Backend = Wgpu;  // NdArray<f32>; LibTorch<f32>; Candle<f32, i64>;

    // Creation of two tensors, the first with explicit values
    // and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &Default::default());
    let tensor_2 = Tensor::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);

    inburn::guide::demo_main()?;
    Ok(())
}

