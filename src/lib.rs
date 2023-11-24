/****************************************************************
 * $ID: lib.rs  	Tue 21 Nov 2023 15:25:27+0800               *
 *                                                              *
 * Maintainer: 范美辉 (MeiHui FAN) <mhfan@ustc.edu>              *
 * Copyright (c) 2023 M.H.Fan, All rights reserved.             *
 ****************************************************************/

pub mod guide;
pub mod mnist;

pub mod mnist_onnx { include!(concat!(env!("OUT_DIR"), "/model/mnist.rs")); }

