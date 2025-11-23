import multiprocessing as mp
import time
import os
import json
import logging

import ray
from pprint import pprint
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from ray.autoscaler.sdk import request_resources
from ray._private.accelerators import TPUAcceleratorManager

from lm_eval.__main__ import setup_parser, parse_eval_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ray.init()

VERBOSE = (os.getenv("VERBOSE", "0") == "1")

TPU_VERSION = os.getenv("TPU_VERSION", "v6e-8")
TPU_POD_COUNT = os.getenv("TPU_POD_COUNT", "1")
TPU_ZONE = os.getenv("TPU_ZONE", "")
PORT = int(os.environ.get("COORD_PORT", "9876"))

WANDB_API_KEY = os.getenv("WANDB_API_KEY_FOR_EASYDEL", "")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", None)

num_slices = int(TPU_POD_COUNT)

base_env = {
    # "JAX_PLATFORMS": "tpu",
    "EASYDEL_PROFILING": os.getenv("EASYDEL_PROFILING", "1"),  # Enable EasyDeL profiling.
    "EASYDEL_PROFILING_DIR": os.getenv("EASYDEL_PROFILING_DIR", f"gs://gidd-checkpoints_{TPU_ZONE[:-2]}_hns/jax-trace"),  # Directory for EasyDeL profiling outputs.
    "EASYDEL_AUTO": os.getenv("EASYDEL_AUTO", "1"),  # Enables EasyDeL's automatic sharding configuration.
    "LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=98304" if "v6e" in TPU_VERSION else "",
    "JAX_LOG_COMPILES": os.getenv("JAX_LOG_COMPILES", "0"),
    "JAX_EXPLAIN_CACHE_MISSES": os.getenv("JAX_EXPLAIN_CACHE_MISSES", "0"),
    "HF_TOKEN": os.getenv("HF_TOKEN_FOR_EASYDEL", ""),  # Hugging Face token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # RAM-disk for dataset cache.
    "HF_HOME": "/dev/shm/huggingface",  # RAM-disk for model cache.
    "HF_DATASETS_OFFLINE": "0",  # Prevent crash if HF is down
    "TRANSFORMERS_OFFLINE": "0",  # Need for tokenizer
    "WANDB_API_KEY": WANDB_API_KEY,  # W&B API key.
    "TPU_NAME": os.getenv("TPU_NAME", ""),
    "TPU_VERSION": TPU_VERSION,
    "TPU_ZONE": TPU_ZONE,
    "TPU_POD_COUNT": TPU_POD_COUNT,
}

if not TPU_ZONE:
    logger.warning("TPU_ZONE is not set. This is fine if you're not running on TPU or if the TPU has environment variables SAVE_DIRECTORY and DATA_FILES.")

parser = setup_parser()
ARGS = parse_eval_args(parser)


def kill_vfio_holders():
    import os, subprocess
    subprocess.run(["bash", "-lc", "command -v lsof >/dev/null 2>&1 || (sudo apt-get update -y && sudo apt-get install -y lsof)"], check=False)
    p = subprocess.run(["bash", "-lc", "lsof -t /dev/vfio/* 2>/dev/null | sort -u"], capture_output=True, text=True)
    pids = [pid for pid in p.stdout.split() if pid.isdigit() and int(pid) != os.getpid()]
    if pids:
        subprocess.run(["bash", "-lc", "kill -9 " + " ".join(pids)], check=False)
    return [int(x) for x in pids]

@ray.remote
def main():
    killed = kill_vfio_holders()
    if killed:
        print(f"Killed {len(killed)} process(es) holding /dev/vfio/*: {killed}")
        time.sleep(5)

    import easydel as ed
    from lm_eval.__main__ import cli_evaluate

    try:
        pprint(ARGS)
        cli_evaluate(ARGS)
    except Exception as e:
        print(f"An error occurred during eval: {e}")
        raise
    else:
        print("Successfully completed eval")

    print("Returning from main()")


def submit_to_host(remote_fn, host_info, env):
    call = remote_fn.options(
        num_cpus=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=host_info["node_id"], soft=False),
        resources={"TPU": host_info["num_tpus"]},
        runtime_env={"env_vars": env},
        max_retries=0,
        retry_exceptions=False,
    ).remote()
    return call


def submit_to_slice(remote_fn, slice_info, slice_id, coord_ip, coord_port=PORT):
    if VERBOSE:
        print(json.dumps(slice_info, indent=2, sort_keys=True))
    hosts_per_slice = slice_info["num_hosts"]
    calls = []
    for host_id in range(hosts_per_slice):
        if num_slices > 1:
            env = dict(base_env,
                MEGASCALE_COORDINATOR_ADDRESS=coord_ip,
                MEGASCALE_NUM_SLICES=str(num_slices),
                MEGASCALE_PORT=str(coord_port),
                MEGASCALE_SLICE_ID=str(slice_id),
            )
        else:
            env = dict(base_env)
        calls.append(
            submit_to_host(remote_fn, slice_info["hosts"][host_id], env)
        )
    return calls


@ray.remote(num_cpus=0)
def discover_hosts_on_slice():
    import ray
    return {
        "ip": ray.util.get_node_ip_address(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "pod_name": ray.util.accelerators.tpu.get_current_pod_name(),
        "num_tpus": TPUAcceleratorManager.get_current_node_num_accelerators(),
    }


@ray.remote(num_cpus=0)
def discover_slice():
    import ray
    pod_name = ray.util.accelerators.tpu.get_current_pod_name()
    num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
    num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
    tpu_type = TPUAcceleratorManager._get_current_node_tpu_pod_type()

    logger.info(f"Discovered slice {pod_name=}: {num_hosts=}, {num_tpus_per_host=}, {tpu_type=}")

    bundles = [{"CPU": 0, pod_name: 1} for _ in range(num_hosts)]
    request_resources(bundles=bundles)
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    try:
        host_info = ray.get([
            discover_hosts_on_slice.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i, True)).remote()
            for i in range(num_hosts)
        ])

        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "ip": ray.util.get_node_ip_address(),
            "pod_name": pod_name,
            "num_hosts": num_hosts,
            "num_tpus_per_host": num_tpus_per_host,
            "tpu_type": tpu_type,
            "hosts": host_info,
        }, pg
    except:
        remove_placement_group(pg)
        raise


def submit_to_multislice(remote_fn, tpu_type, num_slices=1):
    label = f"TPU-{tpu_type}-head"
    bundles = [{"CPU": 0, label: 1} for _ in range(num_slices)]
    request_resources(bundles=bundles)

    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    all_pgs = [pg]

    results = ray.get([
        discover_slice.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i, True)).remote()
        for i in range(num_slices)
    ])
    slice_infos, host_pgs = zip(*results)
    all_pgs.extend(host_pgs)

    logger.info(f"Discovered {len(slice_infos)} slice(s):")
    for si in slice_infos:
        logger.info(f"    + {si['pod_name']} ({si['tpu_type']}, {si['num_hosts']} hosts, {si['num_tpus_per_host']} TPU/host)")


    # logger.info("Waiting for 10s before starting jobs...")
    # time.sleep(10)

    coord_ip = slice_infos[0]['ip']

    # num_procs = sum(
    #     slice_infos[slice_id]["num_hosts"] * slice_infos[slice_id]["num_tpus_per_host"]
    #     for slice_id in range(num_slices)
    # )

    try:
        calls = []
        for slice_id in range(num_slices):
            calls.extend(submit_to_slice(
                remote_fn=remote_fn,
                slice_info=slice_infos[slice_id],
                slice_id=slice_id,
                coord_ip=coord_ip,
                coord_port=PORT,
            ))
        return calls, all_pgs
    except:
        for pg in all_pgs:
            remove_placement_group(pg)
        raise


def run_on_multislice_resumable(
    remote_fn,
    tpu_type,
    num_slices=1,
    max_errors=0,
    max_preemptions=128,
):
    assert WANDB_ENTITY is not None and WANDB_PROJECT is not None, "W&B entity and project must be set for resumable run"
    num_preemptions = 0
    num_errors = 0
    done = False
    for _ in range(max_errors + max_preemptions):
        calls, pgs = [], []
        try:
            calls, pgs = submit_to_multislice(remote_fn, tpu_type, num_slices)
            results = ray.get(calls)
            logger.info("Successfully completed multislice submission")
            done = True
            break
        except (
            ray.exceptions.RayError,
            ray.exceptions.RayTaskError,
            ray.exceptions.TaskUnschedulableError,
        ) as e:
            err_str = str(e).lower()
            logger.info("Caught an error during multislice submission, handling...")
            if any(x in err_str for x in ["preempted", "not schedulable"]):
                num_preemptions += 1
                logger.warning(f"TPU job preempted ({num_preemptions=}): {e}")
                if num_preemptions > max_preemptions:
                    logger.error("Maximum number of preemptions reached. Exiting.")
                    raise
            elif (
                "couldn't connect to 'https://huggingface.co'" in err_str
                or "Read timed out".lower() in err_str
                or "BrokenProcessPool".lower() in err_str
                # or "Unable to initialize backend 'tpu'".lower() in err_str
            ):
                logger.warning(f"TPU job failed for a known reason. Resubmitting without penalty: {e}", exc_info=e)
            else:
                num_errors += 1
                logger.warning(f"TPU job failed ({num_errors=}): {e}", exc_info=e)
                if num_errors > max_errors:
                    logger.error("Maximum number of errors reached. Exiting.")
                    raise

            logger.info("Attempting to resubmit...")
        finally:
            # let's just chill for a bit first
            time.sleep(10)
            logger.info("Canceling remaining tasks...")
            for call in calls:
                try:
                    ray.cancel(call, recursive=True)
                except Exception as e:
                    logger.warning(f"Failed to cancel call {call}: {e}", exc_info=e)
            logger.info("Releasing placement groups...")
            for pg in pgs:
                try:
                    remove_placement_group(pg)
                except Exception as e:
                    logger.warning(f"Failed to remove placement group {pg}: {e}", exc_info=e)
        # and chill again
        logger.info("Waiting 30s before resubmitting...")
        time.sleep(30)

    if done:
        logger.info("All done!")
    else:
        logger.error("Failed to complete multislice submission after retries.")

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('forkserver', force=True)
    run_on_multislice_resumable(
        remote_fn=main,
        tpu_type=TPU_VERSION,
        num_slices=num_slices,
    )
