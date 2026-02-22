
This is a good exercise in getting into the details of object identification. I am going to use this exercise to start getting into more of the GPU hardware. For example, if we run this on Jetson NANO vs. Thor - how will the performance look like?

Can I run this in a cloud environment that has access to more powerful GPUs like the Blackwell?

Simple exercise but than be expanded into more comprehensive vision related items

Explore interactions between Torch, CUBLAS, cuTensor and cuDNN vs. LLMs of today - for example, if I brute force feed images to an LLM and ask it to detect objects etc... what will be the throughput and what will be the performance/cost


------------

## Torch related

a. Why is a warm-up run needed?
b. Forward pass inference and model configuration details - need to be understood better


---------

## Ideas for more research

a. How can we move this to a video for object detection?
b. Compare this to skeleton pose detection algorithms like the google movenet? which can be ran on a CPU