# controlnet-preprocessor
A server for performing the preprocessing steps required for using controlnet with stable diffusion. i.e. generate the normal map, the depth map, etc.

This is a flask server wrapping the [controlnet_aux library](https://github.com/patrickvonplaten/controlnet_aux), which itself wraps the excellent work done by [lllyasviel](https://github.com/lllyasviel)