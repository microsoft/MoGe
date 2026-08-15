import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
from typing import *
import importlib
import importlib.util

import click


def _load_baseline(baseline_code_path: str, extra_args: Sequence[str]) -> 'MGEBaselineInterface':
    """Load the baseline model from its python file, forwarding the unparsed CLI args to its `load` command."""
    from moge.utils.tools import import_file_as_module

    module = import_file_as_module(baseline_code_path, Path(baseline_code_path).stem)
    baseline_cls: Type['MGEBaselineInterface'] = getattr(module, 'Baseline')
    return baseline_cls.load.main(list(extra_args), standalone_mode=False)


def _evaluate_benchmarks(
    baseline: 'MGEBaselineInterface',
    benchmarks: Iterable[Tuple[str, Dict[str, Any]]],
    *,
    metrics_output_path: Union[str, Path],
    dump_dir: Union[str, Path],
    oracle_mode: bool = False,
    mg: Optional[str] = None,
    dump_pred: bool = False,
    dump_gt: bool = False,
    tqdm_position: Optional[int] = None,
    tqdm_prefix: str = '',
) -> Dict[str, Any]:
    """Evaluate `baseline` on `benchmarks` sequentially, returning `{benchmark_name: metrics}` (without `mean`).

    `metrics_output_path` is where this process saves its own (possibly partial) results; `dump_dir` is the
    root for `--dump_pred` / `--dump_gt` outputs and always derives from the user-supplied `--output`.
    """
    # Lazy import
    import  cv2
    import numpy as np
    from tqdm import tqdm
    import torch

    from moge.test.dataloader import EvalDataLoaderPipeline
    from moge.test.metrics import compute_metrics
    from moge.utils.geometry_torch import intrinsics_to_fov
    from moge.utils.vis import colorize_depth, colorize_normal
    from moge.utils.tools import key_average, timeit

    all_metrics = {}
    # A worker claims its benchmarks one at a time, so only the single-process path knows the total upfront.
    if tqdm_position is None:
        benchmarks = tqdm(list(benchmarks), desc='Benchmarks')
    # Iterate over the dataset
    for benchmark_name, benchmark_config in benchmarks:
        metrics_list = []
        with (
            EvalDataLoaderPipeline(**benchmark_config) as eval_data_pipe,
            tqdm(total=len(eval_data_pipe), desc=f'{tqdm_prefix}{benchmark_name}', position=tqdm_position, leave=False) as pbar
        ):
            # Iterate over the samples in the dataset
            for i in range(len(eval_data_pipe)):
                sample = eval_data_pipe.get()
                sample = {k: v.to(baseline.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                image = sample['image']
                gt_intrinsics = sample['intrinsics']

                # Inference
                torch.cuda.synchronize()
                with torch.inference_mode(), timeit('_inference_timer', verbose=False) as timer:
                    if oracle_mode:
                        pred = baseline.infer_for_evaluation(image, gt_intrinsics)
                    else:
                        pred = baseline.infer_for_evaluation(image)
                    torch.cuda.synchronize()

                # Compute metrics
                metrics, misc = compute_metrics(pred, sample, vis=dump_pred or dump_gt, mg=mg)
                metrics['inference_time'] = timer.time
                metrics_list.append(metrics)

                # Dump results
                dump_path = Path(dump_dir, f'{benchmark_name}', sample['filename'].replace('.zip', ''))
                if dump_pred:
                    dump_path.joinpath('pred').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'pred' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    with Path(dump_path, 'pred', 'metrics.json').open('w') as f:
                        json.dump(metrics, f, indent=4)

                    if 'pred_points' in misc:
                        points = misc['pred_points'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'points.exr'), cv2.cvtColor(points.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if 'pred_depth' in misc:
                        depth = misc['pred_depth'].cpu().numpy()
                        if 'mask' in pred:
                            mask = pred['mask'].cpu().numpy()
                            depth = np.where(mask, depth, np.inf)
                        cv2.imwrite(str(dump_path / 'pred' / 'depth.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))

                    if 'mask' in pred:
                        mask = pred['mask'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'mask.png'), (mask * 255).astype(np.uint8))

                    if 'normal' in pred:
                        normal = pred['normal'].cpu().numpy()
                        cv2.imwrite(str(dump_path / 'pred' / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))

                    if 'intrinsics' in pred:
                        intrinsics = pred['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'pred' / 'fov.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)

                if dump_gt:
                    dump_path.joinpath('gt').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dump_path / 'gt' / 'image.jpg'), cv2.cvtColor((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    if 'points' in sample:
                        points = sample['points']
                        cv2.imwrite(str(dump_path / 'gt' / 'points.exr'), cv2.cvtColor(points.cpu().numpy().astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if 'depth' in sample:
                        depth = sample['depth']
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' / 'depth.png'), cv2.cvtColor(colorize_depth(depth.cpu().numpy(), mask=mask.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'normal' in sample:
                        normal = sample['normal']
                        cv2.imwrite(str(dump_path / 'gt' / 'normal.png'), cv2.cvtColor(colorize_normal(normal.cpu().numpy()), cv2.COLOR_RGB2BGR))

                    if 'depth_mask' in sample:
                        mask = sample['depth_mask']
                        cv2.imwrite(str(dump_path / 'gt' /'mask.png'), (mask.cpu().numpy() * 255).astype(np.uint8))

                    if 'intrinsics' in sample:
                        intrinsics = sample['intrinsics']
                        fov_x, fov_y = intrinsics_to_fov(intrinsics)
                        with open(dump_path / 'gt' / 'info.json', 'w') as f:
                            json.dump({
                                'fov_x': np.rad2deg(fov_x.item()),
                                'fov_y': np.rad2deg(fov_y.item()),
                                'intrinsics': intrinsics.cpu().numpy().tolist(),
                            }, f)

                # Save intermediate results
                if i % 100 == 0 or i == len(eval_data_pipe) - 1:
                    Path(metrics_output_path).write_text(
                        json.dumps({
                            **all_metrics,
                            benchmark_name: key_average(metrics_list)
                        }, indent=4)
                    )
                pbar.update(1)

            all_metrics[benchmark_name] = key_average(metrics_list)

    return all_metrics


def _resolve_visible_devices(ngpu: int) -> List[str]:
    """Return `ngpu` device tokens to be assigned to the workers as their `CUDA_VISIBLE_DEVICES`."""
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        # NOTE: entries of CUDA_VISIBLE_DEVICES are absolute ids (and may be GPU UUIDs or MIG ids),
        # so they are sliced as opaque strings and never renumbered.
        devices = [d.strip() for d in cuda_visible_devices.split(',') if d.strip()]
        source = 'CUDA_VISIBLE_DEVICES'
    else:
        import torch      # queried via NVML, does not create a CUDA context
        devices = [str(i) for i in range(torch.cuda.device_count())]
        source = 'torch.cuda.device_count()'

    if not devices:
        raise click.UsageError(f'No CUDA device is available ({source} reports none).')
    if ngpu > len(devices):
        raise click.UsageError(f'--ngpu {ngpu} exceeds the {len(devices)} available GPU(s) ({source}: {",".join(devices)}).')
    return devices[:ngpu]


def _partial_output_path(output_path: Union[str, Path], rank: int) -> Path:
    """`eval_output/moge.json` -> `eval_output/moge.rank0.json`"""
    path = Path(output_path)
    return path.with_name(f'{path.stem}.rank{rank}{path.suffix or ".json"}')


def _claim_benchmarks(benchmarks: Sequence[Tuple[str, Dict[str, Any]]], counter) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield benchmarks claimed one at a time from a counter shared by all workers.

    Whoever finishes first takes the next benchmark, so the workers stay busy no matter how unevenly
    the benchmarks are sized. A plain shared counter is used rather than a queue because a queue's
    feeder thread may not have flushed yet when a worker first polls it, which would make that worker
    give up before any work is visible.
    """
    while True:
        with counter.get_lock():
            index = counter.value
            counter.value += 1
        if index >= len(benchmarks):
            return
        yield benchmarks[index]


def _worker_entry(
    rank: int,
    device: str,
    baseline_code_path: str,
    extra_args: List[str],
    benchmarks: List[Tuple[str, Dict[str, Any]]],
    counter,
    partial_output_path: str,
    dump_dir: str,
    oracle_mode: bool,
    mg: Optional[str],
    dump_pred: bool,
    dump_gt: bool,
    tqdm_lock,
) -> None:
    """Entry point of a single-GPU worker process. Must stay at module level to be picklable by `spawn`."""
    # Pin this worker to its GPU. This MUST happen before anything imports torch, hence the assertion:
    # a future module-level `import torch` would otherwise silently put every worker on the same GPU.
    assert 'torch' not in sys.modules, 'torch was imported before CUDA_VISIBLE_DEVICES could be pinned'
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

    try:
        from tqdm import tqdm
        tqdm.set_lock(tqdm_lock)    # serialize the progress bars across processes

        baseline = _load_baseline(baseline_code_path, extra_args)
        all_metrics = _evaluate_benchmarks(
            baseline, _claim_benchmarks(benchmarks, counter),
            metrics_output_path=partial_output_path,
            dump_dir=dump_dir,
            oracle_mode=oracle_mode,
            mg=mg,
            dump_pred=dump_pred,
            dump_gt=dump_gt,
            tqdm_position=rank,
            tqdm_prefix=f'[gpu {device}] ',
        )
        # Written explicitly instead of relying on the intermediate save, which never fires for a worker
        # that ends up claiming nothing.
        Path(partial_output_path).write_text(json.dumps(all_metrics, indent=4))
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        import traceback
        print(f'\n[rank {rank} | CUDA_VISIBLE_DEVICES={device}] evaluation failed:', file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)


def _run_multi_gpu(
    ngpu: int,
    baseline_code_path: str,
    extra_args: Sequence[str],
    benchmarks: Sequence[Tuple[str, Dict[str, Any]]],
    output_path: Union[str, Path],
    dump_dir: Union[str, Path],
    oracle_mode: bool,
    mg: Optional[str],
    dump_pred: bool,
    dump_gt: bool,
) -> Dict[str, Any]:
    """Distribute the benchmarks over `ngpu` worker processes and merge their results."""
    import multiprocessing as mp

    devices = _resolve_visible_devices(ngpu)
    num_workers = min(ngpu, len(benchmarks))
    if num_workers < ngpu:
        click.echo(f'Note: the config has only {len(benchmarks)} benchmark(s); using {num_workers} of the {ngpu} requested GPUs.', err=True)
    devices = devices[:num_workers]

    partial_paths = [_partial_output_path(output_path, rank) for rank in range(num_workers)]
    for path in partial_paths:
        path.unlink(missing_ok=True)    # never merge a leftover file from a previous run

    # `spawn` (not `fork`, not `torch.multiprocessing.spawn`) gives a fresh interpreter whose first
    # executed statement is ours, which is what makes the CUDA_VISIBLE_DEVICES pinning above possible.
    ctx = mp.get_context('spawn')
    tqdm_lock = ctx.RLock()
    counter = ctx.Value('i', 0)     # index of the next benchmark to be claimed; see `_claim_benchmarks`
    processes = []
    try:
        for rank in range(num_workers):
            process = ctx.Process(
                target=_worker_entry,
                args=(
                    rank, devices[rank], baseline_code_path, list(extra_args), list(benchmarks), counter,
                    str(partial_paths[rank]), str(dump_dir), oracle_mode, mg, dump_pred, dump_gt, tqdm_lock,
                ),
                name=f'eval-rank{rank}-gpu{devices[rank]}',
                daemon=True,    # safe: the workers only ever spawn threads, never processes
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    finally:
        # Leave no orphans behind on Ctrl-C or on a parent-side exception.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
        print('\n' * num_workers, file=sys.stderr, end='')      # move the cursor below the pinned progress bars

    # Collect the per-rank results, treating a crashed worker as a hard failure.
    all_metrics, failures = {}, []
    for rank, (process, partial_path) in enumerate(zip(processes, partial_paths)):
        if process.exitcode != 0:
            failures.append(f'rank {rank} (CUDA_VISIBLE_DEVICES={devices[rank]}) exited with code {process.exitcode}')
            continue
        if not partial_path.exists():
            failures.append(f'rank {rank} (CUDA_VISIBLE_DEVICES={devices[rank]}) produced no result file at {partial_path}')
            continue
        all_metrics.update(json.loads(partial_path.read_text()))

    # Which benchmarks a worker claims is only decided at run time, so completeness is checked globally.
    # This also catches a worker that died in the middle of a benchmark, leaving a valid but truncated snapshot.
    if missing := [benchmark_name for benchmark_name, _ in benchmarks if benchmark_name not in all_metrics]:
        failures.append(f'no result was produced for {missing}')

    if failures:
        kept = [path for path in partial_paths if path.exists()]
        raise click.ClickException(
            'the multi-GPU evaluation did not complete:\n'
            + '\n'.join(f'  {failure}' for failure in failures)
            + f'\n{output_path} was not written.'
            + (
                '\nThe per-rank partial results are kept for inspection:\n' + '\n'.join(f'  {path}' for path in kept)
                if kept else ''
            )
        )

    for path in partial_paths:
        path.unlink(missing_ok=True)
    return all_metrics


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, help='Evaluation script.')
@click.option('--baseline', 'baseline_code_path', type=click.Path(), required=True, help='Path to the baseline model python code.')
@click.option('--config', 'config_path', type=click.Path(), default='configs/eval/all_benchmarks.json', help='Path to the evaluation configurations. '
    'Defaults to "configs/eval/all_benchmarks.json".')
@click.option('--output', '-o', 'output_path',  type=click.Path(), required=True, help='Path to the output json file.')
@click.option('--ngpu', 'ngpu', type=click.IntRange(min=1), default=1, help='Number of GPUs to use. Whole benchmarks of the config are '
    'distributed over one worker process per GPU, each claiming the next benchmark as it goes. Defaults to 1, i.e. a single in-process run.')
@click.option('--oracle', 'oracle_mode', is_flag=True, help='Use oracle mode for evaluation, i.e., use the GT intrinsics input.')
@click.option('--mg', 'mg', type=str, default='moge3', help='Comma-separated metric groups to compute.')
@click.option('--dump_pred', is_flag=True, help='Dump predition results.')
@click.option('--dump_gt', is_flag=True, help='Dump ground truth.')
@click.pass_context
def main(ctx: click.Context, baseline_code_path: str, config_path: str, ngpu: int, oracle_mode: bool, mg: Optional[str], output_path: Union[str, Path], dump_pred: bool, dump_gt: bool):
    from moge.utils.tools import key_average      # a stdlib-only module: importing it keeps the parent torch-free

    # Load the evaluation configurations
    with open(config_path, 'r') as f:
        config = json.load(f)
    benchmarks = list(config.items())
    if not benchmarks:
        raise click.UsageError(f'No benchmark is configured in {config_path}.')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(str(output_path).replace('.json', '_dump'))

    if ngpu == 1 or len(benchmarks) == 1:
        # Single benchmark: the worker count would be capped to 1 anyway, and that worker would get the
        # very same GPU as an in-process run, so skip the subprocess entirely.
        baseline = _load_baseline(baseline_code_path, ctx.args)
        all_metrics = _evaluate_benchmarks(
            baseline, benchmarks,
            metrics_output_path=output_path,
            dump_dir=dump_dir,
            oracle_mode=oracle_mode,
            mg=mg,
            dump_pred=dump_pred,
            dump_gt=dump_gt,
        )
    else:
        if any(arg == '--device' or arg.startswith('--device=') for arg in ctx.args):
            raise click.UsageError(
                '--device cannot be combined with --ngpu > 1: every worker is pinned to its own GPU via '
                'CUDA_VISIBLE_DEVICES, and an explicit --device would send all of them to the same physical GPU. '
                'Drop --device, or use --ngpu 1.'
            )
        per_benchmark_metrics = _run_multi_gpu(
            ngpu, baseline_code_path, ctx.args, benchmarks, output_path, dump_dir,
            oracle_mode, mg, dump_pred, dump_gt,
        )
        # Restore the config order, which depends on which worker happened to claim what.
        all_metrics = {benchmark_name: per_benchmark_metrics[benchmark_name] for benchmark_name, _ in benchmarks}

    # Save final results
    all_metrics['mean'] = key_average(list(all_metrics.values()))
    Path(output_path).write_text(json.dumps(all_metrics, indent=4))


if __name__ == '__main__':
    main()
