from flask import Flask, request, make_response, jsonify, send_file
from PIL import Image
from io import BytesIO
import os
import logging
import time
from waitress import serve
from rembg import remove
from PIL import Image
from controlnet_aux import (
    HEDdetector,
    MidasDetector,
    MLSDdetector,
    OpenposeDetector,
    PidiNetDetector,
    NormalBaeDetector,
    LineartDetector,
    LineartAnimeDetector,
    CannyDetector,
    ContentShuffleDetector,
    # ZoeDetector,
    MediapipeFaceDetector,
    SamDetector,
    LeresDetector,
)
from __version__ import VERSION

log = logging.getLogger()
log.setLevel(logging.INFO)

log.info("Version: ", VERSION)

# Load config from the environment
host = os.environ.get("HOST", "localhost")
port = int(os.environ.get("PORT", 2222))
annotator_path = os.environ.get("ANNOTATOR_PATH", "/models/lllyasviel/Annotators")
sam_path = os.environ.get("SAM_PATH", "/models/ybelkada/segment-anything")
sam_subfolder = os.environ.get("SAM_SUBFOLDER", "checkpoints")

app = Flask(__name__)

# load checkpoints
log.info("Loading checkpoints...")
hed = HEDdetector.from_pretrained(annotator_path)
midas = MidasDetector.from_pretrained(annotator_path)
mlsd = MLSDdetector.from_pretrained(annotator_path)
open_pose = OpenposeDetector.from_pretrained(annotator_path)
pidi = PidiNetDetector.from_pretrained(annotator_path)
normal_bae = NormalBaeDetector.from_pretrained(annotator_path)
lineart = LineartDetector.from_pretrained(annotator_path)
lineart_anime = LineartAnimeDetector.from_pretrained(annotator_path)
# zoe = ZoeDetector.from_pretrained(annotator_path)
sam = SamDetector.from_pretrained(os.path.join(sam_path, sam_subfolder))
leres = LeresDetector.from_pretrained(annotator_path)

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()


def remove_bg(image_bytes, output_type="pil"):
    # Remove bg and return as PIL or bytes as specified
    result = remove(image_bytes)
    return result


log.info("Checkpoints loaded.")

processors = {
    "scribble_hed": {"callable": hed, "config": {"scribble": True}},
    "softedge_hed": {"callable": hed, "config": {"scribble": False}},
    "scribble_hedsafe": {
        "callable": hed,
        "config": {"scribble": True, "safe": True},
    },
    "softedge_hedsafe": {
        "callable": hed,
        "config": {"scribble": False, "safe": True},
    },
    "depth_midas": {"callable": midas, "config": {}},
    "mlsd": {"callable": mlsd, "config": {}},
    "openpose": {
        "callable": open_pose,
        "config": {"include_body": True, "include_hand": False, "include_face": False},
    },
    "openpose_face": {
        "callable": open_pose,
        "config": {"include_body": True, "include_hand": False, "include_face": True},
    },
    "openpose_faceonly": {
        "callable": open_pose,
        "config": {"include_body": False, "include_hand": False, "include_face": True},
    },
    "openpose_full": {
        "callable": open_pose,
        "config": {"include_body": True, "include_hand": True, "include_face": True},
    },
    "openpose_hand": {
        "callable": open_pose,
        "config": {"include_body": False, "include_hand": True, "include_face": False},
    },
    "scribble_pidinet": {
        "callable": pidi,
        "config": {"safe": False, "scribble": True},
    },
    "softedge_pidinet": {
        "callable": pidi,
        "config": {"safe": False, "scribble": False},
    },
    "scribble_pidsafe": {
        "callable": pidi,
        "config": {"safe": True, "scribble": True},
    },
    "softedge_pidsafe": {
        "callable": pidi,
        "config": {"safe": True, "scribble": False},
    },
    "normal_bae": {"callable": normal_bae, "config": {}},
    "lineart_realistic": {"callable": lineart, "config": {"coarse": False}},
    "lineart_coarse": {"callable": lineart, "config": {"coarse": True}},
    "lineart_anime": {"callable": lineart_anime, "config": {}},
    "canny": {"callable": canny, "config": {}},
    "shuffle": {"callable": content, "config": {}},
    # "depth_zoe": {"callable": zoe, "config": {}},
    "depth_leres": {"callable": leres, "config": {"boost": False}},
    "depth_leres++": {"callable": leres, "config": {"boost": True}},
    "mediapipe_face": {"callable": face_detector, "config": {}},
    "sam": {"callable": sam, "config": {}},
    "remove_background": {"callable": remove_bg, "config": {}},
}


@app.get("/hc")
def hc():
    log.info({"version": VERSION})
    return make_response(jsonify({"version": VERSION}), 200)


@app.get("/processors")
def get_processors():
    # return just the processor ids
    return make_response(jsonify({"processors": list(processors.keys())}), 200)


@app.post("/image/<processor_id>")
def process_image(processor_id: str):
    request_start = time.perf_counter()
    # Get the image from the request
    try:
        image = Image.open(BytesIO(request.data)).convert("RGB")
        # Get the original width and height of the image
        width, height = image.size

        # Calculate the new dimensions while preserving aspect ratio
        if width > height:
            new_width = min(width, 1024)
            new_height = int(height * new_width / width)
        else:
            new_height = min(height, 1024)
            new_width = int(width * new_height / height)

        # Resize the image
        image = image.resize((new_width, new_height))
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 400)

    # Check if the processor is allowed
    if processor_id not in processors:
        return make_response(
            jsonify({"error": f"Processor {processor_id} not found"}), 400
        )

    # Process the image
    processor = processors[processor_id]["callable"]
    config = processors[processor_id]["config"]
    inference_start = time.perf_counter()
    try:
        result = processor(image, output_type="pil", **config)
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)
    inference_end = time.perf_counter()
    # Return the result as a lossless webp
    buffer = BytesIO()
    try:
        if processor_id == "remove_background":
            export = {"format": "png"}
        else:
            export = {"format": "webp", "lossless": True}
        result.save(buffer, **export)
        buffer.seek(0)
        request_end = time.perf_counter()
        response = make_response(
            send_file(buffer, mimetype=f"image/{export['format']}")
        )
        response.headers["X-Request-Time"] = str(request_end - request_start)
        response.headers["X-Inference-Time"] = str(inference_end - inference_start)
        return response
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)


if __name__ == "__main__":
    serve(app, host=host, port=port, ipv6=True)
