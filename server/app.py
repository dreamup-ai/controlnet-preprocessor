from flask import Flask, request, make_response, jsonify, send_file
from PIL import Image
from io import BytesIO
import os
import logging
from waitress import serve
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
    ZoeDetector,
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
port = os.environ.get("PORT", 2222)
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
zoe = ZoeDetector.from_pretrained(annotator_path)
sam = SamDetector.from_pretrained(os.path.join(sam_path, sam_subfolder))
leres = LeresDetector.from_pretrained(annotator_path)

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()

log.info("Checkpoints loaded.")

processors = {
    "scribble_hed": {"class": hed, "config": {"scribble": True}},
    "softedge_hed": {"class": hed, "config": {"scribble": False}},
    "scribble_hedsafe": {
        "class": hed,
        "config": {"scribble": True, "safe": True},
    },
    "softedge_hedsafe": {
        "class": hed,
        "config": {"scribble": False, "safe": True},
    },
    "depth_midas": {"class": midas, "config": {}},
    "mlsd": {"class": mlsd, "config": {}},
    "open_pose": {
        "class": open_pose,
        "config": {"include_body": True, "include_hand": False, "include_face": False},
    },
    "open_pose_face": {
        "class": open_pose,
        "config": {"include_body": True, "include_hand": False, "include_face": True},
    },
    "open_pose_faceonly": {
        "class": open_pose,
        "config": {"include_body": False, "include_hand": False, "include_face": True},
    },
    "open_pose_full": {
        "class": open_pose,
        "config": {"include_body": True, "include_hand": True, "include_face": True},
    },
    "open_pose_hand": {
        "class": open_pose,
        "config": {"include_body": False, "include_hand": True, "include_face": False},
    },
    "scribble_pidinet": {
        "class": pidi,
        "config": {"safe": False, "scribble": True},
    },
    "softedge_pidinet": {
        "class": pidi,
        "config": {"safe": False, "scribble": False},
    },
    "scribble_pidsafe": {
        "class": pidi,
        "config": {"safe": True, "scribble": True},
    },
    "softedge_pidsafe": {
        "class": pidi,
        "config": {"safe": True, "scribble": False},
    },
    "normal_bae": {"class": normal_bae, "config": {}},
    "lineart_realistic": {"class": lineart, "config": {"coarse": False}},
    "lineart_coarse": {"class": lineart, "config": {"coarse": True}},
    "lineart_anime": {"class": lineart_anime, "config": {}},
    "canny": {"class": canny, "config": {}},
    "shuffle": {"class": content, "config": {}},
    "depth_zoe": {"class": zoe, "config": {}},
    "depth_leres": {"class": leres, "config": {"boost": False}},
    "depth_leres++": {"class": leres, "config": {"boost": True}},
    "mediapipe_face": {"class": face_detector, "config": {}},
    "sam": {"class": sam, "config": {}},
}


@app.get("/hc")
def hc():
    log.info({"version": VERSION})
    return make_response(jsonify({"version": VERSION}), 200)


@app.post("/image/<processor_id>")
def process_image(processor_id: str):
    # Get the image from the request
    try:
        image = Image.open(BytesIO(request.data)).convert("RGB").resize((512, 512))
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 400)

    # Check if the processor is allowed
    if processor_id not in processors:
        return make_response(
            jsonify({"error": f"Processor {processor_id} not found"}), 400
        )

    # Process the image
    processor = processors[processor_id]["class"]
    config = processors[processor_id]["config"]
    try:
        result = processor(image, output_type="pil", **config)
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)

    # Return the result as a lossless webp
    buffer = BytesIO()
    try:
        result.save(buffer, format="webp", lossless=True)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/webp")
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)


if __name__ == "__main__":
    serve(app, host=host, port=port)
