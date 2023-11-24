
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{tensor::Tensor, backend::Wgpu as Backend};

    // Creation of two tensors, the first with explicit values
    // and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);

    use inburn::mnist::training::train;
    use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
    //use burn::backend::{NdArrayBackend, ndarray::NdArrayDevice};

    let artifact_dir = "data/mnist";
    train::<Autodiff<Wgpu>>(artifact_dir, WgpuDevice::default())?;
    //train::<Autodiff<NdArrayBackend>>(artifact_dir, NdArrayDevice::Cpu)?;

    Ok(())
}

