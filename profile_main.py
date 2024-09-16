import psutil
import subprocess
import numpy as np
import time

def compile_program():
    subprocess.run(["cargo", "build", "--release"])

def run_program():
    executable = "target/release/main"
    input_file = "input.txt"

    t_start = time.time()
    process = subprocess.Popen([executable], stdin=open(input_file, "r"), stdout=subprocess.PIPE)

    # Monitor memory usage
    memory_peak = 0
    while process.poll() is None:
        memory_current = psutil.Process(process.pid).memory_info().rss
        memory_peak = max(memory_peak, memory_current)
        time.sleep(0.01)  # Sleep for a short time to reduce CPU usage

    t_end = time.time()
    t_elapsed = t_end - t_start
    
    # Get the output and error (if any)
    stdout, stderr = process.communicate()

    return {
        "time": t_elapsed,
        "memory": memory_peak,
        "stdout": stdout,
        "stderr": stderr
    }


def main():

    # run program n_sample=10 times
    # and get the statistics (mean +- std, min, max)

    n_sample = 10
    times = []
    memories = []
    for i in range(n_sample):
        print(f"Running sample {i+1}/{n_sample}")
        result = run_program()
        times.append(result["time"])
        memories.append(result["memory"])
    times = np.array(times)
    memories = np.array(memories)


    p = lambda x: f"{x:.4f}"
    print(f"Time: {p(times.mean())} +- {p(times.std())} s")

    p = lambda x: f"{x/1e6:.2f}"
    print(f"Peak memory: {p(memories.mean())} +- {p(memories.std())} MB")

if __name__ == "__main__":
    main()