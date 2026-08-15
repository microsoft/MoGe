# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Donwload the processed datasets from [Huggingface Datasets](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and put them in the `data/eval` directory, using `huggingface-cli`:

```bash
mkdir -p data/eval
huggingface-cli download Ruicheng/monocular-geometry-evaluation --repo-type dataset --local-dir data/eval --local-dir-use-symlinks False
```

Then unzip the downloaded files:

```bash
cd data/eval  
unzip '*.zip'
# rm *.zip # if you don't keep the zip files
```

## Configuration

See [`configs/eval/moge3.json`](../configs/eval/moge3.json) for an example of evaluation configurations on all benchmarks. You can modify this file to evaluate on different benchmarks or different baselines.

Each entry maps a benchmark name to a config object. Supported keys:

| Key | Default | Description |
| --- | --- | --- |
| `path` | *required* | Root directory of the processed dataset. |
| `width`, `height` | *required* | Target evaluation resolution. |
| `split` | `.index.txt` | Index file listing sample directories, relative to `path`. |
| `depth_unit` | `null` | Scale factor applied to GT depth. **Setting this marks the benchmark as metric**; leaving it out disables all `*_metric` metric groups. |
| `depth` | `depth.png` | Depth map filename within each sample directory. |
| `has_sharp_boundary` | `false` | Enables the boundary F1 metrics. |
| `segmentation` | `null` | Segmentation map filename. Required by the `points_local_moge2` group. |
| `normal` | `null` | Normal map filename. |
| `local_mask` | `null` | Binary mask filename marking the local detail region. Required by the `local` metric groups. |
| `local_segmentation` | `null` | SAM segment-id map filename. Intersected with `local_mask` to recover per-segment regions. Only read when `local_mask` is also set. |
| `drop_max_depth` | `1000.` | Drop depth beyond this multiple of the 1% depth quantile. |
| `max_segments`, `min_seg_area` | `100`, `1000` | Segment filtering for `points_local_moge2`. |
| `subset` | `null` | Take every N-th sample. Useful for quick smoke runs. |


## Baseline

Some examples of baselines are provided in [`baselines/`](../baselines/). Pass the path to the baseline model python code to the `--baseline` argument of the evaluation script. 

## Metric groups

Use `--mg` to choose which metrics to compute, which is significantly faster when you only care about a few metrics. `--mg` takes a comma-separated list of **suites**, **categories**, or **concrete group names**.

There are two named suites:

| Suite | Contents |
| --- | --- |
| `moge3` | Default setting. The metric set reported by MoGe-3: `global` + `metric` + `local` + `boundary_f1_r1` |
| `moge2` | The metric set reported by MoGe-2: `global` + `metric` + `points_local_moge2` + `boundary_f1_r123` |

The suites are built from these categories, which can also be requested directly:

| Category | Expands to |
| --- | --- |
| `global` | `depth_affine_invariant`, `depth_scale_invariant`, `disparity_affine_invariant`, `points_affine_invariant`, `points_scale_invariant`, `fov_x` |
| `metric` | `depth_metric`, `points_metric` for metric benchmarks |
| `local` | `depth_local`, `points_local` |


## Run Evaluation

Run the script [`moge/scripts/eval_baseline.py`](../moge/scripts/eval_baseline.py). 
For example, 

```bash
# Evaluate MoGe-3 on the 10 benchmarks with 3 refine steps
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/moge3.json --output eval_output/moge.json --pretrained PATH_TO_CKPT.pt --resolution_level 9 --version v3 --refine_steps 3

# Same as the first one, but spread over 4 GPUs (see "Multi-GPU Evaluation" below)
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/moge3.json --output eval_output/moge.json --ngpu 4 --pretrained PATH_TO_CKPT.pt --resolution_level 9 --version v3 --refine_steps 3

# Evaluate MoGe on the 10 benchmarks
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/moge2.json --output eval_output/moge.json --pretrained Ruicheng/moge-vitl --resolution_level 9

# Evaluate Depth Anything V2 on the 10 benchmarks. (NOTE: affine disparity)
python moge/scripts/eval_baseline.py --baseline baselines/da_v2.py --config configs/eval/moge2.json --output eval_output/da_v2.json

# Only global metrics, skipping the expensive local and boundary ones
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/moge3.json --output eval_output/moge.json --mg global --pretrained PATH_TO_CKPT.pt --version v3

# The metric set of MoGe-2, for comparison against older results
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/moge3.json --output eval_output/moge.json --mg moge2 --pretrained PATH_TO_CKPT.pt --version v3
```

The `--baseline` `--input` `--output` arguments are for the inference script. The rest arguments, e.g. `--pretrained` `--resolution_level`, are custormized for loading the baseline model.

Details of the arguments:

```
Usage: eval_baseline.py [OPTIONS]

  Evaluation script.

Options:
  --baseline PATH       Path to the baseline model python code.  [required]
  --config PATH         Path to the evaluation configurations. Defaults to
                        "configs/eval/all_benchmarks.json".
  -o, --output PATH     Path to the output json file.  [required]
  --ngpu INTEGER RANGE  Number of GPUs to use. Whole benchmarks of the config
                        are distributed over one worker process per GPU, each
                        claiming the next benchmark as it goes. Defaults to 1,
                        i.e. a single in-process run.  [x>=1]
  --oracle              Use oracle mode for evaluation, i.e., use the GT
                        intrinsics input.
  --mg TEXT             Comma-separated metric groups to compute.
  --dump_pred           Dump predition results.
  --dump_gt             Dump ground truth.
  --help                Show this message and exit.
```


## Multi-GPU Evaluation

`--ngpu N` spawns one worker process per GPU and hands each worker whole benchmarks of the config. Whoever finishes first takes the next unclaimed benchmark, so the GPUs stay busy despite the benchmarks being very unevenly sized.

```bash
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/moge3.json --output eval_output/moge.json --ngpu 4 --pretrained PATH_TO_CKPT.pt --version v3
```



## Wrap a Customized Baseline

Wrap any baseline method with [`moge.test.baseline.MGEBaselineInterface`](../moge/test/baseline.py).
See [`baselines/`](../baselines/) for more examples.

It is a good idea to check the correctness of the baseline implementation by running inference on a small set of images via [`moge/scripts/infer_baselines.py`](../moge/scripts/infer_baselines.py):

```base
python moge/scripts/infer_baselines.py --baseline baselines/moge.py --input example_images/ --output infer_outupt/moge --pretrained Ruicheng/moge-vitl --maps --ply
```


