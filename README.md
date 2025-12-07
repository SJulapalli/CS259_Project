Run the project from within build-local.

make sure to run cmake and make to set up llama-cli

The way I usually run it is ./bin/llama-cli -p "<Your prompt here>" -m model.gguf -b batch_size(necessary for hmt style segmentation) -c context_length(must fit prompt) -n num_output_tokens

You can similarly quantize a model with llama-quantize.

Getting a model into .gguf format can be done with convert_hf_to_gguf.py if you have a safetensors file in HMT format. In order to properly match the expected format, you need to have your file merge
the lora layers into a single layer. This can be done by running main in HMT-pytorch with the same setup we're used to for accelerate, which will output the safetensors file. I pretty much just commented everything
that actually has to do with running/training the model out, so it'll just set up HMT and the save it in safetensors format.
