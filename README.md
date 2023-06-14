# controlnet-preprocessor
A server for performing the preprocessing steps required for using controlnet with stable diffusion. i.e. generate the normal map, the depth map, etc.

This is a containerized flask server wrapping the [controlnet_aux library](https://github.com/patrickvonplaten/controlnet_aux), which itself wraps the excellent work done by [lllyasviel](https://github.com/lllyasviel)

It is available with and without the models baked into the container.

# API

## GET /hc

Health check endpoint. Returns 200 with the server version if the server is up.

## GET /processors

Returns a list of available processors.

### cURL

```bash
curl  -X GET \
  'http://localhost:2222/processors'
```

### Response

```json
{
  "processors": [
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
    "depth_leres",
    "depth_leres++",
    "mediapipe_face",
    "sam"
  ]
}
```

## POST /image/\<processor-id>

Upload an image, and get the processed, annotated image back. The image you receive back is what you should pass to controlnet as the source image.

### cURL

```bash
curl  -X POST \
  'http://localhost:2222/image/<processor-id>' \
  --header 'Content-Type: application/octet-stream' \
  --data-binary '@/path/to/image.png'
```