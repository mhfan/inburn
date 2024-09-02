/****************************************************************
 * $ID: guide.rs  	Tue 21 Nov 2023 09:19:10+0800               *
 *                                                              *
 * Maintainer: 范美辉 (MeiHui FAN) <mhfan@ustc.edu>              *
 * Copyright (c) 2023 M.H.Fan, All rights reserved.             *
 ****************************************************************/

pub mod mnist {     //  https://github.com/Tracel-AI/burn/tree/main/examples/mnist

use burn::{module::Module, tensor::{backend::Backend, Tensor},
    nn::{self, BatchNorm, PaddingConfig2d}, config::Config,
};

#[derive(Module, Debug)] pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    conv3: ConvBlock<B>,
    dropout: nn::Dropout,
    fc1:  nn::Linear<B>,
    fc2:  nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Default for Model<B> { fn default() -> Self { Self::new(&B::Device::default()) } }

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = ConvBlock::new([ 1,  8], [3, 3], device); // out: [Batch,8,26,26]
        let conv2 = ConvBlock::new([ 8, 16], [3, 3], device); // out: [Batch,16,24x24]
        let conv3 = ConvBlock::new([16, 24], [3, 3], device); // out: [Batch,24,22x22]
        let fc1 = nn::LinearConfig::new(24 * 22 * 22, 32) // 16 * 8 * 8
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

#[derive(Config, Debug)] pub struct ModelConfig {
    #[config(default = "0.5")] dropout: f64,
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {  /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> { Model::new(device) }
}

}

pub mod model {     //  https://github.com/Tracel-AI/burn/tree/main/examples/guide

use burn::{ module::Module, tensor::{backend::Backend, Tensor}, config::Config,
    nn::{conv::{Conv2d, Conv2dConfig}, Dropout, DropoutConfig, Linear, LinearConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig}, Relu,
    },
};

//  https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#[derive(Module, Debug)] pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    /// ## Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();
        let x = images.reshape([batch_size, 1, height, width]);
        // Create a channel at the second dimension.

        let x = self.conv1.forward(x); // [batch_size,  8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);  // [batch_size, 16, 8, 8]
        //let [batch_size, channels, height, width] = x.dims();
        //let x = x.reshape([batch_size, channels * height * width]);
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

impl ModelConfig {  /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> { Model {
        conv1: Conv2dConfig::new([ 1,  8], [3, 3]).init(device),
        conv2: Conv2dConfig::new([ 8, 16], [3, 3]).init(device),
        pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(), activation: Relu::new(),
        linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
        linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        dropout: DropoutConfig::new(self.dropout).init(),
    } }
}

}

pub mod data {

use burn::{tensor::{backend::Backend, ElementConversion, TensorData, Int, Tensor},
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
};

#[derive(Clone)] pub struct MnistBatcher<B: Backend> { device: B::Device, }

impl<B: Backend> MnistBatcher<B> { pub fn new(device: B::Device) -> Self { Self { device } } }

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items.iter().map(|item|
            TensorData::from(item.image)).map(|data|
            Tensor::<B, 2>::from_data(data, &self.device)).map(|tensor|
            tensor.reshape([1, 28, 28])).map(|tensor|
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are copied from PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f45725/mnist/main.py#L122
            ((tensor / 255) - 0.1307) / 0.3081).collect();

        let targets = items.iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(
                [(item.label as i32).elem::<B::IntElem>()], &self.device)).collect();

        MnistBatch { images: Tensor::cat( images, 0).to_device(&self.device),
                    targets: Tensor::cat(targets, 0).to_device(&self.device) }
    }
}

#[derive(Clone, Debug)] pub struct MnistBatch<B: Backend> {
    pub targets: Tensor<B, 1, Int>,
    pub  images: Tensor<B, 3>,
}

}

//use model::{Model, ModelConfig};
use mnist::{Model, ModelConfig};

pub mod training {

use super::{Model, ModelConfig, data::{MnistBatch, MnistBatcher}};
use burn::{config::Config, module::Module, tensor::backend::{AutodiffBackend, Backend},
    nn::loss::CrossEntropyLossConfig, optim::AdamConfig,
    record::{CompactRecorder, NoStdTrainingRecorder},
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        MetricEarlyStoppingStrategy, StoppingCondition::NoImprovementSince,
        metric::{AccuracyMetric, LossMetric, CpuUse, CpuMemory, CpuTemperature,
            store::{Aggregate, Direction, Split},
        }
    },
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device()).forward(output.clone(), targets.clone());
        ClassificationOutput { loss, output, targets, }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>,  ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

#[derive(Config)] pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 42)] pub seed: u64,
    #[config(default = 10)] pub  num_epochs: usize,
    #[config(default = 64)] pub  batch_size: usize,
    #[config(default =  4)] pub num_workers: usize,
    #[config(default = 1e-4)] pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig,
    device: B::Device) -> Result<(), Box<dyn std::error::Error>> {
    // Remove existing artifacts before to get an accurate learner summary
    //std::fs::remove_dir_all(artifact_dir)?;

    // TODO: check and load config/module, for increment training?
    std::fs::create_dir_all(artifact_dir).and_then(|_|
        config.save(format!("{artifact_dir}/config.json")))?;
    B::seed(config.seed);

    let dataloader_train = DataLoaderBuilder::new(
        MnistBatcher::<B>::new(device.clone()))
        .batch_size(config.batch_size).shuffle(config.seed)
        .num_workers(config.num_workers).build(MnistDataset::train());

    let dataloader_valid = DataLoaderBuilder::new(
        MnistBatcher::<B::InnerBackend>::new(device.clone()))
        .batch_size(config.batch_size).shuffle(config.seed)
        .num_workers(config.num_workers).build(MnistDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(Aggregate::Mean,
            Direction::Lowest, Split::Valid, NoImprovementSince { n_epochs: 1 },
        )).devices(vec![device.clone()])
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs).summary()
        .build(config.model.init::<B>(&device), config.optimizer.init(), config.learning_rate);

    learner.fit(dataloader_train, dataloader_valid).save_file(format!("{artifact_dir}/model"),
        &NoStdTrainingRecorder::new())?;  Ok(())  // &CompactRecorder::new()
}

}

use burn::{tensor::backend::Backend, data::dataset::vision::MnistItem};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) ->
    Result<(), Box<dyn std::error::Error>> {
    use self::{data::MnistBatcher, training::TrainingConfig};
    use burn::{module::Module, config::Config, data::dataloader::batcher::Batcher,
        record::{Recorder, NoStdTrainingRecorder} //CompactRecorder
    };

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))?;
    let record = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)?;    //CompactRecorder::new()
    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batch = MnistBatcher::new(device).batch(vec![item]);
    let predicted = model.forward(batch.images)
        .argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);     Ok(())
}

//#[cfg(test)] mod tests { }

pub fn demo_main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{backend::{Autodiff, wgpu::Wgpu, //ndarray::NdArray,
        }, data::dataset::{Dataset, vision::MnistDataset},
        optim::{AdamConfig, decay::WeightDecayConfig},
    };  use training::{train, TrainingConfig};

    //&format!("{}/guide", env!("OUT_DIR"));
    let (dev, dir) = (Default::default(), "target/guide");
    let mcfg = ModelConfig::new(10, 512);
    println!("{}", mcfg.init::<Wgpu>(&dev));

    let tcfg = TrainingConfig::new(mcfg,
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))));
    train::<Autodiff<Wgpu>>(dir, tcfg, dev.clone())?;

    infer::<Wgpu>(dir, dev, MnistDataset::test().get(42)
        .ok_or("Fail to get test data from MNIST dataset")?)?;

    Ok(())
}

