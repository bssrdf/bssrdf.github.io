# A Basic Training Example Using ggml

I just want to share what I have been working on recently. This is an [example](https://github.com/bssrdf/ggml/blob/one-adam-opt-call/examples/vae/mnist-vae2.cpp) of training a MNIST VAE. The goal is to use only `ggml` pipeline and its implementation of ADAM optimizer. 

There aren't many training examples using `ggml`. ~~The only one I found is [baby-llama](https://github.com/xaedes/llama.cpp/blob/train-example/examples/baby-llama/baby-llama.cpp). But I think its way of doing opmization is not quite right.~~ Found another [training example ](https://github.com/ggerganov/llama.cpp/tree/master/examples/train-text-from-scratch) in `llama.cpp` which shows a proper way of using Adam.

## Some of the mods I have to add
-  Reuse the same forward and backward graph during training
-  Change in Adam and LBFGS optimizer to make GPU backend work
-  Add several missing OPs in both CPU and CUDA backends 
-  Hooks (callbacks) added in optimizer to do tests and sample work
 
Below are some samples from the VAE trained on MNIST after each epoch (total 10 epochs).

![mnist-sample-epoch_1](https://github.com/ggerganov/ggml/assets/689043/26837c4d-7b2c-4e97-af04-19e3fc9ed28e) | ![mnist-sample-epoch_2](https://github.com/ggerganov/ggml/assets/689043/1518df66-ab9f-47ec-8a05-90e3d5dbee0d)
![mnist-sample-epoch_3](https://github.com/ggerganov/ggml/assets/689043/3184945c-42d0-4940-88cb-6db3278e6cfd) | ![mnist-sample-epoch_4](https://github.com/ggerganov/ggml/assets/689043/e42e0d25-4b2c-4b45-b1de-5869bfb9aea7)
![mnist-sample-epoch_5](https://github.com/ggerganov/ggml/assets/689043/f6b19fd7-c0e0-46b9-8812-9cfa35290a34) | ![mnist-sample-epoch_6](https://github.com/ggerganov/ggml/assets/689043/b5e4e64b-0647-48d8-8440-033b5df1ed2c)
![mnist-sample-epoch_7](https://github.com/ggerganov/ggml/assets/689043/e848006d-663e-4d45-8e3c-f5aa03c7befe) | ![mnist-sample-epoch_8](https://github.com/ggerganov/ggml/assets/689043/73c634d6-f57a-477a-865f-7727c250a881)
![mnist-sample-epoch_9](https://github.com/ggerganov/ggml/assets/689043/8ad7d194-f39c-40a5-85a2-25b7efb2603a) | ![mnist-sample-epoch_10](https://github.com/ggerganov/ggml/assets/689043/141c8a89-d077-42e4-b1d1-4f9f1b930447)
