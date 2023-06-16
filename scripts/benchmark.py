import requests
import os
import time
import json
import random
import string

base_url = os.getenv("BASE_URL", "http://localhost:1234/")
if base_url[-1] != "/":
    base_url += "/"

# Load the image
image_path = os.path.abspath(os.getenv("IMAGE_PATH", "images/original.png"))
image = open(image_path, "rb").read()

output_dir = os.getenv("OUTPUT_DIR", "benchmark_output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

api_key = os.getenv("API_KEY", "")
api_header = os.getenv("API_HEADER", "Salad-Api-Key")

processors = [
    "scribble_hed",
    "softedge_hed",
    "scribble_hedsafe",
    "softedge_hedsafe",
    "depth_midas",
    "mlsd",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "scribble_pidinet",
    "softedge_pidinet",
    "scribble_pidsafe",
    "softedge_pidsafe",
    "normal_bae",
    "lineart_realistic",
    "lineart_coarse",
    "lineart_anime",
    "canny",
    "shuffle",
    # "depth_zoe",
    "depth_leres",
    "depth_leres++",
    "mediapipe_face",
    "sam",
]


def randomSuffix(numChars=4):
    return "".join(random.choice(string.ascii_letters) for i in range(numChars))


batchId = randomSuffix()

output_file = os.getenv("OUTPUT_FILE", f"data-{batchId}.json")

# For every processor id in processors, we're going to upload an image to the server and time the response
# We'll do this 10 times and take the average
all_data = {}
for processor_id in processors:
    url = base_url + "/image/" + processor_id
    print("Testing", processor_id)
    times = []
    for i in range(10):
        start = time.perf_counter()
        if api_key:
            response = requests.post(url, data=image, headers={api_header: api_key})
        else:
            response = requests.post(url, data=image)
        # inference time and internal request time are included in headers
        inference_time = response.headers["X-Inference-Time"]
        internal_request_time = response.headers["X-Request-Time"]
        end = time.perf_counter()
        round_trip_time = end - start
        times.append(
            {
                "round_trip_time": float(round_trip_time),
                "inference_time": float(inference_time),
                "internal_request_time": float(internal_request_time),
            }
        )

    avg_round_trip_time = sum([t["round_trip_time"] for t in times]) / len(times)
    avg_inference_time = sum([t["inference_time"] for t in times]) / len(times)
    avg_internal_request_time = sum([t["internal_request_time"] for t in times]) / len(
        times
    )
    min_round_trip_time = min([t["round_trip_time"] for t in times])
    min_inference_time = min([t["inference_time"] for t in times])
    min_internal_request_time = min([t["internal_request_time"] for t in times])
    max_round_trip_time = max([t["round_trip_time"] for t in times])
    max_inference_time = max([t["inference_time"] for t in times])
    max_internal_request_time = max([t["internal_request_time"] for t in times])
    all_data[processor_id] = {
        "data": times,
        "round_trip": {
            "avg": avg_round_trip_time,
            "min": min_round_trip_time,
            "max": max_round_trip_time,
        },
        "inference": {
            "avg": avg_inference_time,
            "min": min_inference_time,
            "max": max_inference_time,
        },
        "internal_request": {
            "avg": avg_internal_request_time,
            "min": min_internal_request_time,
            "max": max_internal_request_time,
        },
    }
    print(
        f"Average round trip time: {avg_round_trip_time:.3f}s, min: {min_round_trip_time:.3f}s, max: {max_round_trip_time:.3f}s"
    )
    print(
        f"Average inference time: {avg_inference_time:.3f}s, min: {min_inference_time:.3f}s, max: {max_inference_time:.3f}s"
    )
    print(
        f"Average internal request time: {avg_internal_request_time:.3f}s, min: {min_internal_request_time:.3f}s, max: {max_internal_request_time:.3f}s"
    )

# Save the data
with open(os.path.join(output_dir, output_file), "w") as f:
    print(json.dumps(all_data, indent=2))
    json.dump(all_data, f)
    print("Saved data to", f.name)
