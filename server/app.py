from flask import Flask, request, Response
from PIL import Image
from io import BytesIO
import os
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

# Load config from the environment
host = os.environ.get("HOST", "localhost")
port = os.environ.get("PORT", 2222)
annotator_path = os.environ.get("ANNOTATOR_PATH", "/models/lllyasviel/Annotators")
sam_path = os.environ.get("SAM_PATH", "/models/ybelkada/segment-anything")
sam_subfolder = os.environ.get("SAM_SUBFOLDER", "checkpoints")

app = Flask(__name__)

# load checkpoints
hed = HEDdetector.from_pretrained(annotator_path)
midas = MidasDetector.from_pretrained(annotator_path)
mlsd = MLSDdetector.from_pretrained(annotator_path)
open_pose = OpenposeDetector.from_pretrained(annotator_path)
pidi = PidiNetDetector.from_pretrained(annotator_path)
normal_bae = NormalBaeDetector.from_pretrained(annotator_path)
lineart = LineartDetector.from_pretrained(annotator_path)
lineart_anime = LineartAnimeDetector.from_pretrained(annotator_path)
zoe = ZoeDetector.from_pretrained(annotator_path)
sam = SamDetector.from_pretrained(sam_path, subfolder=sam_subfolder)
leres = LeresDetector.from_pretrained(annotator_path)

# instantiate
canny = CannyDetector()
content = ContentShuffleDetector()
face_detector = MediapipeFaceDetector()

processors = {
    "hed": {"class": HEDdetector, "config": {"scribble": True}},
    "softedge_hed": {"class": HEDdetector, "config": {"scribble": False}},
    "scribble_hedsafe": {
        "class": HEDdetector,
        "config": {"scribble": True, "safe": True},
    },
    "softedge_hedsafe": {
        "class": HEDdetector,
        "config": {"scribble": False, "safe": True},
    },
    "depth_midas": {"class": MidasDetector, "config": {}},
    "mlsd": {"class": MLSDdetector, "config": {}},
    "open_pose": {
        "class": OpenposeDetector,
        "config": {"include_body": True, "include_hand": False, "include_face": False},
    },
    "open_pose_face": {
        "class": OpenposeDetector,
        "config": {"include_body": True, "include_hand": False, "include_face": True},
    },
    "open_pose_faceonly": {
        "class": OpenposeDetector,
        "config": {"include_body": False, "include_hand": False, "include_face": True},
    },
    "open_pose_full": {
        "class": OpenposeDetector,
        "config": {"include_body": True, "include_hand": True, "include_face": True},
    },
    "open_pose_hand": {
        "class": OpenposeDetector,
        "config": {"include_body": False, "include_hand": True, "include_face": False},
    },
    "scribble_pidinet": {
        "class": PidiNetDetector,
        "config": {"safe": False, "scribble": True},
    },
    "softedge_pidinet": {
        "class": PidiNetDetector,
        "config": {"safe": False, "scribble": False},
    },
    "scribble_pidsafe": {
        "class": PidiNetDetector,
        "config": {"safe": True, "scribble": True},
    },
    "softedge_pidsafe": {
        "class": PidiNetDetector,
        "config": {"safe": True, "scribble": False},
    },
    "normal_bae": {"class": NormalBaeDetector, "config": {}},
    "lineart_realistic": {"class": LineartDetector, "config": {"coarse": False}},
    "lineart_coarse": {"class": LineartDetector, "config": {"coarse": True}},
    "lineart_anime": {"class": LineartAnimeDetector, "config": {}},
    "canny": {"class": CannyDetector, "config": {}},
    "shuffle": {"class": ContentShuffleDetector, "config": {}},
    "depth_zoe": {"class": ZoeDetector, "config": {}},
    "depth_leres": {"class": LeresDetector, "config": {"boost": False}},
    "depth_leres++": {"class": LeresDetector, "config": {"boost": True}},
    "mediapipe_face": {"class": MediapipeFaceDetector, "config": {}},
}


@app.get("/hc")
def hc():
    return Response({"version": VERSION}, status=200, mimetype="application/json")


@app.post("/image/<processor_id>")
def process_image(processor_id: str):
    # Get the image from the request
    try:
        image = Image.open(BytesIO(request.data)).convert("RGB").resize((1024, 1024))
    except Exception as e:
        return Response({"error": str(e)}, status=400, mimetype="application/json")

    # Check if the processor is allowed
    if processor_id not in processors:
        return Response(
            {"error": f"Processor {processor_id} not found"},
            status=400,
            mimetype="application/json",
        )

    # Process the image
    processor = processors[processor_id]["class"]
    config = processors[processor_id]["config"]
    try:
        result = processor(image, **config)
    except Exception as e:
        return Response({"error": str(e)}, status=500, mimetype="application/json")

    # Return the result as a lossless webp
    buffer = BytesIO()
    try:
        result.save(buffer, format="webp", lossless=True)
        buffer.seek(0)
        return Response(buffer, status=200, mimetype="image/webp")
    except Exception as e:
        return Response({"error": str(e)}, status=500, mimetype="application/json")


if __name__ == "__main__":
    serve(app, host=host, port=port)
