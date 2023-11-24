/****************************************************************
 * $ID: guide.rs  	Tue 21 Nov 2023 09:19:10+0800               *
 *                                                              *
 * Maintainer: 范美辉 (MeiHui FAN) <mhfan@ustc.edu>              *
 * Copyright (c) 2023 M.H.Fan, All rights reserved.             *
 ****************************************************************/

//  https://burn.dev/book/, https://github.com/Tracel-AI/burn/tree/main/examples/guide

pub mod model {

use burn::{config::Config, module::Module,
    nn::{conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
    },  tensor::{backend::Backend, Tensor},
};

//  https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#[derive(Module, Debug)] pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    //conv3: ConvBlock<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    pool: AdaptiveAvgPool2d,
    activation: ReLU,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size,  8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);  // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}

#[derive(Config, Debug)] pub struct ModelConfig {
    #[config(default = "0.5")] dropout: f64,
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self) -> Model<B> { Model { // Returns the initialized model.
        conv1: Conv2dConfig::new([1,  8], [3, 3]).init(),
        conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
        linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
        linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
        pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(), activation: ReLU::new(),
        dropout: DropoutConfig::new(self.dropout).init(),
    } }

    /// Returns the initialized model using the recorded weights.
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> { Model {
        conv1: Conv2dConfig::new([1,  8], [3, 3]).init_with(record.conv1),
        conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
        linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
        linear2: LinearConfig::new(self.hidden_size, self.num_classes).init_with(record.linear2),
        pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(), activation: ReLU::new(),
        dropout: DropoutConfig::new(self.dropout).init(),
    } }
}

}

pub mod data {

use burn::{tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
};

pub struct MNISTBatcher<B: Backend> { device: B::Device, }

impl<B: Backend> MNISTBatcher<B> { pub fn new(device: B::Device) -> Self { Self { device } } }

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items.iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081).collect();

        let targets = items.iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(
                Data::from([(item.label as i64).elem()]))).collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MNISTBatch { images, targets }
    }
}

#[derive(Clone, Debug)] pub struct MNISTBatch<B: Backend> {
    pub targets: Tensor<B, 1, Int>,
    pub  images: Tensor<B, 3>,
}

}

pub mod training {

use super::{model::{Model, ModelConfig}, data::{MNISTBatch, MNISTBatcher}};
use burn::{self, config::Config, module::Module,
    tensor::{Int, Tensor, backend::{AutodiffBackend, Backend}},
    nn::loss::CrossEntropyLoss, record::CompactRecorder, optim::AdamConfig,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    train::{metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep},
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(&self, images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::default()
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images,
            batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)] pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 42)] pub seed: u64,
    #[config(default = 10)] pub  num_epochs: usize,
    #[config(default = 64)] pub  batch_size: usize,
    #[config(default =  4)] pub num_workers: usize,
    #[config(default = 1.0e-4)] pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device)
    -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(artifact_dir).and_then(|_|
        config.save(format!("{artifact_dir}/config.json")))?;
    B::seed(config.seed);

    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(MNISTBatcher::<B>::new(device.clone()))
        .batch_size(config.batch_size).shuffle(config.seed)
        .num_workers(config.num_workers).build(MNISTDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size).shuffle(config.seed)
        .num_workers(config.num_workers).build(MNISTDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device]).num_epochs(config.num_epochs)
        .build(config.model.init::<B>(), config.optimizer.init(), config.learning_rate);

    learner.fit(dataloader_train, dataloader_test).save_file(
        format!("{artifact_dir}/model"), &CompactRecorder::new())?;     Ok(())
}

}

use burn::{tensor::backend::Backend, data::dataset::source::huggingface::MNISTItem};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem)
    -> Result<(), Box<dyn std::error::Error>> {
    use self::{data::MNISTBatcher, training::TrainingConfig};
    use burn::{config::Config, module::Module, data::dataloader::batcher::Batcher,
        record::{CompactRecorder, Recorder},
    };

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))?;
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())?;
    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batch = MNISTBatcher::new(device).batch(vec![item]);
    let predicted = model.forward(batch.images)
        .argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);     Ok(())
}

#[cfg(test)] mod tests {

#[test] fn demo_main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{backend::{wgpu::WgpuDevice, Wgpu, Autodiff},
        data::dataset::{Dataset, source::huggingface::MNISTDataset}, optim::AdamConfig};
    use super::{model::ModelConfig, training::{train, TrainingConfig}, infer};

    let (artifact_dir, device) = ("data/guide", WgpuDevice::default());
    train::<Autodiff<Wgpu>>(artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()), device.clone())?;

    infer::<Wgpu>(artifact_dir, device, MNISTDataset::test().get(42)
        .ok_or("Fail to get test data from MNIST dataset")?)?;

    Ok(())
}

}

