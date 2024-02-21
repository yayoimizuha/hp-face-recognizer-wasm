mod utils;

use std::f32;
use std::convert::TryFrom;
use std::ops::{Add, Div, Mul};
use itertools::{enumerate, iproduct, Itertools};
use ndarray::{arr1, arr2, Array, array, Array2, Axis, concatenate, Ix1, Ix2, Ix3, s};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

fn prior_box(min_sizes: Vec<Vec<usize>>, steps: Vec<usize>, clip: bool, image_size: [usize; 2]) -> (Array2<f32>, usize) {
    let feature_maps = steps.iter().map(|&step| {
        [f32::ceil(image_size[0] as f32 / step as f32) as i32,
            f32::ceil(image_size[1] as f32 / step as f32) as i32]
    }).collect::<Vec<_>>();
    // println!("{:?}", feature_maps);
    let mut anchors: Vec<[f32; 4]> = vec![];
    for (k, f) in enumerate(feature_maps) {
        for (i, j) in iproduct!(0..f[0],0..f[1]) {
            let t_min_sizes = &min_sizes[k];
            for &min_size in t_min_sizes {
                let s_kx = min_size as f32 / image_size[1] as f32;
                let s_ky = min_size as f32 / image_size[0] as f32;
                let dense_cx = [j as f32 + 0.5].iter().map(|x| x * steps[k] as f32 / image_size[1] as f32).collect::<Vec<_>>();
                let dense_cy = [i as f32 + 0.5].iter().map(|y| y * steps[k] as f32 / image_size[0] as f32).collect::<Vec<_>>();
                for (cy, cx) in iproduct!(dense_cy,dense_cx) {
                    anchors.push([cx, cy, s_kx, s_ky]);
                }
            }
        }
    }
    let mut output = arr2(&anchors);
    if clip {
        output = output.mapv(|x| f32::min(f32::max(x, 0.0), 1.0));
    }
    (output, anchors.len())
}

fn decode(loc: Array<f32, Ix2>, priors: Array<f32, Ix2>, variances: [f32; 2]) -> Array<f32, Ix2> {
    let mut boxes = concatenate(Axis(1), &*vec![
        (priors.slice(s![..,..2]).to_owned() + loc.slice(s![..,..2]).mul(variances[0]) * priors.slice(s![..,2..])).view(),
        (priors.slice(s![..,2..]).to_owned() * loc.slice(s![..,2..]).mul(variances[1]).to_owned().mapv(f32::exp)).view(),
    ]).unwrap();


    let boxes_sub = boxes.slice(s![..,..2]).to_owned() - boxes.slice(s![..,2..]).div(2.0);
    boxes.slice_mut(s![..,..2]).assign(&boxes_sub);

    let boxes_add = boxes.slice(s![..,2..]).to_owned() + boxes.slice(s![..,..2]);
    boxes.slice_mut(s![..,2..]).assign(&boxes_add);
    boxes
}

fn nms_impl(boxes: Array<f32, Ix2>, scores: Array<f32, Ix1>, nms_threshold: f32) -> Vec<usize> {
    let x1 = boxes.slice(s![..,0]).to_owned();
    let y1 = boxes.slice(s![..,1]).to_owned();
    let x2 = boxes.slice(s![..,2]).to_owned();
    let y2 = boxes.slice(s![..,3]).to_owned();

    let areas = (x2.clone() - x1.clone()).add(1.) * (y2.clone() - y1.clone()).add(1.);
    let mut order = scores.iter().enumerate().sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap()).map(|x| x.0).collect::<Vec<_>>();

    let mut keep = vec![];

    let np_maximum = |_x1: f32, _x2: Array<f32, Ix1>| -> Array<f32, Ix1> {
        _x2.mapv(|_x2_val| if _x2_val > _x1 { _x2_val } else { _x1 })
    };
    let np_minimum = |_x1: f32, _x2: Array<f32, Ix1>| -> Array<f32, Ix1> {
        _x2.mapv(|_x2_val| if _x2_val < _x1 { _x2_val } else { _x1 })
    };
    while order.len() > 0 {
        let i = order[0];
        keep.push(i);

        let xx1 = np_maximum(x1[[i]], x1.select(Axis(0), &order[1..]));
        let yy1 = np_maximum(y1[[i]], y1.select(Axis(0), &order[1..]));
        let xx2 = np_minimum(x2[[i]], x2.select(Axis(0), &order[1..]));
        let yy2 = np_minimum(y2[[i]], y2.select(Axis(0), &order[1..]));

        let w = np_maximum(0.0, (xx2 - xx1).add(1.));
        let h = np_maximum(0.0, (yy2 - yy1).add(1.));

        let inter = w * h;
        let ovr = inter.clone() / (areas.select(Axis(0), &order[1..]).add(areas[[i]]) - inter.clone());

        let indices = ovr.iter().enumerate().filter(|(_, val)| val < &&nms_threshold).map(|(order, _)| order).collect::<Vec<_>>();

        order = arr1(&*order).select(Axis(0), &*indices.iter().map(|x| x + 1).collect::<Vec<_>>()).to_vec();
    }
    keep
}

fn decode_landmark(pre: Array<f32, Ix2>, priors: Array<f32, Ix2>, variances: [f32; 2]) -> Array<f32, Ix2> {
    return concatenate(Axis(1),
                       &*vec![
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,..2]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,2..4]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,4..6]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,6..8]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,8..10]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                       ]).unwrap();
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FoundFace {
    bbox: [f32; 4],
    score: f32,
    landmarks: [[f32; 2]; 5],
}

#[wasm_bindgen]
pub async fn post_process(_confidence: Box<[f32]>, _loc: Box<[f32]>, _landmark: Box<[f32]>, scale: f32, height: usize,width:usize) -> Result<String, String> {
    let confidence_threshold = 0.02;
    let nms_threshold = 0.4;
    let vis_threshold = 0.8;
    // let keep_top_k = 750;
    let top_k = 5000;
    let variance = [0.1, 0.2];

    let input_shape = [1, 3, height, width];


    let transformed_size = array![input_shape[2], input_shape[3]].to_owned();
    let scale_landmarks = concatenate(Axis(0), &*vec![transformed_size.view(); 5]).unwrap().mapv(|x| x as f32);
    let scale_bboxes = concatenate(Axis(0), &*vec![transformed_size.view(); 2]).unwrap().mapv(|x| x as f32);

    let (prior_box, onnx_output_width) = prior_box(
        vec![vec![16, 32], vec![64, 128], vec![256, 512]],
        [8, 16, 32].into(),
        false,
        [input_shape[2], input_shape[3]],
    );
    let extract = |tensor: Box<[f32]>, width: usize| <Array<f32, Ix3>>::from_shape_vec((1usize, onnx_output_width, width), Vec::from(tensor)).unwrap();

    let confidence = extract(_confidence, 2);
    let loc = extract(_loc, 4);
    let landmark = extract(_landmark, 10);

    let mut boxes = decode(loc.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
    boxes = boxes * scale_bboxes;
    let mut scores = confidence.slice(s![0,..,1]).to_owned() as Array<f32, Ix1>;
    let mut landmarks = decode_landmark(landmark.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
    landmarks = landmarks * scale_landmarks;

    let indices = scores.iter().enumerate().filter(|(_, val)| val > &&confidence_threshold).map(|(order, _)| order).collect::<Vec<_>>();
    boxes = boxes.select(Axis(0), &*indices);
    landmarks = landmarks.select(Axis(0), &*indices);
    scores = scores.select(Axis(0), &*indices);

    let mut order = scores.clone().iter().enumerate().sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap()).map(|x| x.0).collect::<Vec<_>>();
    if order.len() > top_k {
        order = order[..top_k].to_vec()
    }
    boxes = boxes.select(Axis(0), &*order);
    landmarks = landmarks.select(Axis(0), &*order);
    scores = scores.select(Axis(0), &*order);
    // println!("{:?}", boxes.dim());
    // println!("{:?}", landmarks.dim());
    // println!("{:?}", scores.dim());

    let keep = nms_impl(boxes.clone(), scores.clone(), nms_threshold);

    // println!("{:?}", keep);

    let boxes = boxes.select(Axis(0), &*keep);//.slice(s![..keep_top_k,..]).to_owned();
    let scores = scores.select(Axis(0), &*keep);//.slice(s![..keep_top_k]).to_owned();
    let landmarks = landmarks.select(Axis(0), &*keep);//.slice(s![..keep_top_k,..]).to_owned();

    let vis_score_keep = scores.iter().enumerate().filter(|x| x.1 > &vis_threshold).map(|x| x.0).collect::<Vec<_>>();
    // println!("{:?}", vis_score_keep);


    // let keep = nms(&boxes, &scores.mapv(|x| x as f64), nms_threshold.into(), confidence_threshold).into_iter().collect::<Vec<_>>();


    let mut faces = vec![];
    for index in vis_score_keep {
        faces.push(FoundFace {
            bbox: <[f32; 4]>::try_from(boxes.slice(s![index,..]).mul(scale).to_vec()).unwrap(),
            score: *scores.get(index).unwrap(),
            landmarks: <[[f32; 2]; 5]>::try_from(landmarks.slice(s![index,..]).mul(scale).to_vec().chunks_exact(2).map(|x| { <[f32; 2]>::try_from(x).unwrap() }).collect::<Vec<_>>()).unwrap(),
        });
        // print!("{}\t", boxes.slice(s![index,..]));
        // print!("{}\t", scores.slice(s![index]));
        // println!("{}", landmarks.slice(s![index,..]));
    }


    Ok(serde_json::to_string(&faces).unwrap())
}

