/****************************************************************
 * $ID: mnist.rs  	Tue 21 Nov 2023 15:27:08+0800               *
 *                                                              *
 * Maintainer: 范美辉 (MeiHui FAN) <mhfan@ustc.edu>              *
 * Copyright (c) 2023 M.H.Fan, All rights reserved.             *
 ****************************************************************/

//  https://github.com/Tracel-AI/burn/tree/main/examples/mnist

pub mod model {

use super::data::MNISTBatch;
use burn::{module::Module, tensor::{Tensor, backend::{AutodiffBackend, Backend}},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
    nn::{self, loss::CrossEntropyLossConfig, BatchNorm, PaddingConfig2d},
};

#[derive(Module, Debug)] pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    conv3: ConvBlock<B>,
    dropout: nn::Dropout,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Default for Model<B> { fn default() -> Self { Self::new(&B::Device::default()) } }

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = ConvBlock::new([ 1,  8], [3, 3], device); // out: [Batch,8,26,26]
        let conv2 = ConvBlock::new([ 8, 16], [3, 3], device); // out: [Batch,16,24x24]
        let conv3 = ConvBlock::new([16, 24], [3, 3], device); // out: [Batch,24,22x22]
        let fc1 = nn::LinearConfig::new(24 * 32 * 22, 32)
            .with_bias(false).init(device);
        let fc2 = nn::LinearConfig::new(32, 10).with_bias(false).init(device);
        let dropout = nn::DropoutConfig::new(0.5).init();

        Self { conv1, conv2, conv3, dropout, fc1, fc2, activation: nn::Gelu::new(), }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        let x = self.dropout.forward(x);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        self.fc2.forward(x)
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new().init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput { loss, output, targets, }
    }
}

#[derive(Module, Debug)] pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: nn::Gelu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Valid).init(device);
        let norm = nn::BatchNormConfig::new(channels[1]).init(device);
        Self { conv, norm, activation: nn::Gelu::new(), }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        self.activation.forward(x)
    }
}

impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>,  ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

}

pub mod data {

use burn::{tensor::{backend::Backend, ElementConversion, Data, Int, Tensor},
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
};

#[derive(Clone)] pub struct MNISTBatcher<B: Backend> { device: B::Device, }

impl<B: Backend> MNISTBatcher<B> { pub fn new(device: B::Device) -> Self { Self { device } } }

impl<B: Backend> Batcher<MnistItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MNISTBatch<B> {
        let images = items.iter().map(|item|
            Data::<f32, 2>::from(item.image)).map(|data|
            Tensor::<B, 2>::from_data(data.convert(), &self.device)).map(|tensor|
            tensor.reshape([1, 28, 28])).map(|tensor|
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            ((tensor / 255) - 0.1307) / 0.3081).collect();

        let targets = items.iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(
                Data::from([(item.label as i64).elem()]), &self.device)).collect();

        MNISTBatch { images: Tensor::cat( images, 0).to_device(&self.device),
                    targets: Tensor::cat(targets, 0).to_device(&self.device)}
    }
}

#[derive(Clone, Debug)] pub struct MNISTBatch<B: Backend> {
    pub targets: Tensor<B, 1, Int>,
    pub  images: Tensor<B, 3>,
}

}

pub mod training {

use super::{model::Model, data::MNISTBatcher};
use burn::{config::Config, module::Module, tensor::backend::AutodiffBackend,
    record::{CompactRecorder, NoStdTrainingRecorder},
    optim::{AdamConfig, decay::WeightDecayConfig},
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    train::{MetricEarlyStoppingStrategy, StoppingCondition, LearnerBuilder,
        metric::{CpuUse, CpuMemory, CpuTemperature,
            LossMetric, AccuracyMetric, store::{Aggregate, Direction, Split}}
    },
};

#[derive(Config)] pub struct MnistTrainingConfig {
    #[config(default = 10)] pub  num_epochs: usize,
    #[config(default = 64)] pub  batch_size: usize,
    #[config(default =  4)] pub num_workers: usize,
    #[config(default = 42)] pub seed: u64,
    pub optimizer: AdamConfig,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device)
    -> Result<(), Box<dyn std::error::Error>> {
    // Config
    let config = MnistTrainingConfig::new(AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-5))));
    // TODO: check and load config/module, for increment training?
    std::fs::create_dir_all(artifact_dir).and_then(|_|
        config.save(format!("{artifact_dir}/config.json")))?;
    B::seed(config.seed);

    // Data
    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(MNISTBatcher::<B>::new(device.clone()))
        .batch_size(config.batch_size).shuffle(config.seed)
        .num_workers(config.num_workers).build(MnistDataset::train());
    let dataloader_test  = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size).shuffle(config.seed)
        .num_workers(config.num_workers).build(MnistDataset::test());

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(Aggregate::Mean,
            Direction::Lowest, Split::Valid, StoppingCondition::NoImprovementSince { n_epochs: 1 },
        )).devices(vec![device.clone()]).num_epochs(config.num_epochs)
        .build(Model::new(&device), config.optimizer.init(), 1e-4);

    learner.fit(dataloader_train, dataloader_test).save_file(format!("{artifact_dir}/model"),
        &NoStdTrainingRecorder::new())?;    Ok(())
}

}

