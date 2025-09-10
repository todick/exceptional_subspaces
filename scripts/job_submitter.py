import argparse
import os
import sys
import time
sys.path.insert(0, "./") 
from src.loaders.datasets import get_casestudy_names

USERNAME = os.getenv("USER")
SYFLOW_PATH = f"/scratch/{USERNAME}/analysis/syflow"
LOG_DIR = f"{SYFLOW_PATH}/logs"
JOB_DIR = f"{SYFLOW_PATH}/jobs"
JOB_ID = int(time.time() * 1000)
SBATCH_OPTIONS = f"--constraint=kine --ntasks-per-core=1 --mem=6000 --output={LOG_DIR}/{JOB_ID}/%j.log"
EXECUTE_SCRIPT = f"/scratch/{USERNAME}/analysis/syflow/scripts/execute_job.sh"

def init_files(filename, jobs):
    os.makedirs(f"{LOG_DIR}/{JOB_ID}", exist_ok=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for job in jobs:
            f.write(job + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('dataset', type=str, choices=["distance_based", "workload"])
    parser.add_argument('-m', '--method', type=str, default="syflow", choices=["dfs-mean", "beam-kl", "beam-mean", "rsd", "syflow", "cart"])
    parser.add_argument('-r', '--repetitions', type=int, default=100)
    parser.add_argument('-t', '--time', type=str, default="0-01:00:00")
    parser.add_argument('-c', '--casestudies', type=str, default=None)
    args = parser.parse_args()

    filename = f"{JOB_DIR}/{args.experiment}_{JOB_ID}.txt"    
    script_path = f"experiments/{args.experiment}.py"
    job_prefix = f"cd {SYFLOW_PATH} && source venv/bin/activate && python "
    jobs = []
    for casestudy in get_casestudy_names(args.dataset):
        if args.casestudies is not None and casestudy not in args.casestudies.split(","):
            continue
        for seed in range(args.repetitions):
            jobs.append(f"{job_prefix} {script_path} {args.dataset} {casestudy} --seed {seed} --method {args.method}") 
    init_files(filename, jobs)

    print(f"Submitting {len(jobs)} jobs for experiment {args.experiment} with method {args.method} on dataset {args.dataset}")
    SBATCH_OPTIONS += f" --job-name={args.method}_{args.experiment} --array=1-{len(jobs)} --time={args.time}"
    command = f"sbatch {SBATCH_OPTIONS} {EXECUTE_SCRIPT} {filename}"
    os.system(command)