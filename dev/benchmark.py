import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
from tqdm import tqdm
import requests
import io


def worker(img, idx, main_url):
    start = time.time()
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    buffered.seek(0)

    response = requests.post(
        main_url,
        files={"image": ("image.jpg", buffered, "image/jpeg")},
        timeout=30,
    )
    elapsed = time.time() - start

    ok = response is not None and response.status_code == 200

    try:
        if response.headers.get("content-type", "").startswith("application/json"):
            result = response.json()
            if not result or ("error" in result):
                print(f"API error response: {result.get('error', 'No message')}")
                return idx, elapsed, False, None, None, None, None, None

            return (
                idx,
                elapsed,
                ok,
                result.get("detector_time"),
                result.get("alignment_time"),
                result.get("recognition_time"),
                result.get("search_time"),
                result.get("total_time"),
            )
        else:
            print(f"Response not JSON: {response.content}")
            return idx, elapsed, False, None, None, None, None, None
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return idx, elapsed, False, None, None, None, None, None


def print_stats(arr, name):
    if arr:
        arr = np.array(arr)
        print(
            f"{name} Time:   avg={arr.mean():.3f}s, "
            f"p95={np.percentile(arr, 95):.3f}s, "
            f"min={arr.min():.3f}s, max={arr.max():.3f}s"
        )
    else:
        print(f"{name} Time:   No data")



def run_benchmark(
    n_requests=10000,
    concurrency=100,
    image_path="dev/tests/data/neymar.jpg",
    main_url="http://localhost:8004/benchmark",
):
    img = Image.open(image_path).convert("RGB")

    print("Warming up...")
    for _ in range(3):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)
        _ = requests.post(
            main_url,
            files={"image": ("image.jpg", buffered, "image/jpeg")},
            timeout=30,
        )
    print("Warmup done\n")

    print(f"Running benchmark: {n_requests} requests, concurrency={concurrency}")

    times = []
    success = 0
    detector_times = []
    align_times = []
    recognition_times = []
    search_times = []
    total_times = []

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(worker, img, i, main_url) for i in range(n_requests)
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            idx, elapsed, ok, d_time, a_time, r_time, s_time, t_time = f.result()
            times.append(elapsed)
            if d_time is not None:
                detector_times.append(d_time)
            if a_time is not None:
                align_times.append(a_time)
            if r_time is not None:
                recognition_times.append(r_time)
            if s_time is not None:
                search_times.append(s_time)
            if t_time is not None:
                total_times.append(t_time)
            if ok:
                success += 1

    wall_time = time.time() - start_all

    print("\n========== Benchmark Result ==========")
    print(f"Total requests:     {n_requests}")
    print(f"Success:            {success}")
    print(f"Failed:             {n_requests - success}")
    print(f"Total wall time:    {wall_time:.3f}s")
    print(f"Avg per request:    {np.mean(times):.3f}s")
    print(f"P95 latency:        {np.percentile(times, 95):.3f}s")
    print(f"Min/Max:            {np.min(times):.3f}s / {np.max(times):.3f}s")
    print(f"Throughput (RPS):   {n_requests / wall_time:.1f} req/s")
    print_stats(detector_times, "Detector")
    print_stats(align_times, "Alignment")
    print_stats(recognition_times, "Recognition")
    print_stats(search_times, "Search")
    print_stats(total_times, "Total")
    print("======================================\n")


if __name__ == "__main__":
    run_benchmark(
        n_requests=10000,
        concurrency=100,
        image_path="dev/tests/data/neymar.jpg",
        main_url="http://localhost:8004/benchmark",
    )
