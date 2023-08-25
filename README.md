## Dataset Collection Pipelines for Surrogates

This subdirectory contains files for the dataset collection which was utilized to train the surrogates. The dataset includes architectures randomly sampled from the MnasNet search space, their accuracies, and their device-specific performance metrics such as throughput on all devices, and latency on the FPGA devices. For details on the search-space utilized in this benchmark, please see Appendix. B of the paper. The architectures in the dataset are sampled randomly from the search space. We collected this dataset for roughly 5.2k architectures.

* Accuracy Collection on ImageNet (Proxified)
* Throughput/Latency Collection
	* On FPGAs (Inference throughput and latency using Xilinx DPUs)
	* On TPUs (Inference throughput only)
	* On GPUs (Inference throughput only)

## Architecture, Accuracy Pairs using Training Proxies
We collected a total of ~5k architecture and accuracy pairs on the ImageNet dataset using a host of proxies that reduce the training time to around ~3 GPU-hours per architecture, but incur a substantial accuracy degradation (around 10%) relative to a state-of-the-art training schemes that fully utilize augmentations, regularizations and other fancy training recipes. **A non-exhaustive list of proxies is as follows:**
* [FFCV](https://github.com/libffcv/ffcv-imagenet/tree/main) dataloader with JPEG data compression. We compress the ImageNet dataset using parameters 1) 400px side length, 2) 100% JPEG encoded, and 3) Quality of 90 JPEG. Please see [FFCV ImageNet configurations](https://github.com/libffcv/ffcv-imagenet/tree/main) for details. 
* Only 16 epochs of training per model.
* Batch size of 512 per GPU.
* Progressive resizing similar to FFCV: 160px inputs for the first 12 epochs of training, then 192px until the end of training.
* For detailed configurations, please see config file [here](https://github.com/afzalxo/ANB-DatasetCollection/blob/master/configs/conf_local.cfg).

##### Hardware Setup: 
We utilized 6 to 7 nodes on a SLURM cluster with 4 RTX 3090 GPUs per node. Since FFCV stores the dataset in memory, we recommend at least 64 GiB of RAM on each node for fast dataloading. Our nodes were fitted with 128 - 256 GiB of RAM each. Each node was responsible for training a subset of the total architectures, i.e., roughly 850 architectures per node. The dataset collection pipeline can hence be parallelized by using a large number of nodes. In our case, the accuracy dataset collection took around one month of continuous training on 24 - 28 GPUs. The scripts provided have to be run separately on each node. The `num_models` argument allows setting how many architectures the script will train.

##### Software Requirements: 
For dataset collection and experiment logging, we heavily rely on [wandb](wandb.ai). This is necessary given the huge compute cost of the project and task parallelization across multiple nodes, we needed to ensure that logging can be handled reliably, in a central location. We save each model's weights, training configurations, loss/accuracy curves, accuracy results, FLOPs/#Params metrics, and various other aspects of training on wandb as artifacts. After model training is finished on all the nodes, we use a script to obtain the architecture accuracy pairs from wandb for further processing. 

---
To begin the dataset collection on a node with 4 GPUs, please run the following commands. The commands can be trivially adopted to run on nodes with more or less than 4 GPUs but the hyperparameters, such as learning rate and batch size, have to tuned appropriately. See config file [here](https://github.com/afzalxo/ANB-DatasetCollection/blob/master/configs/conf_local.cfg) for the hyperparameters utilized with our hardware configuration:

First generate FFCV ImageNet dataset using FFCV:
To generate the FFCV ImageNet dataset, use the command: `./write_imagenet.sh 400 1.0 90` in FFCV ImageNet repository [here](https://github.com/libffcv/ffcv-imagenet). This would generate train and validation FFCV format imagenet dataset. Please point the `train_dataset` variable [here](https://github.com/afzalxo/ANB-DatasetCollection/blob/master/configs/conf_local.cfg) to the `train_400_0.50_90.ffcv` file that you generated, and `val_dataset` variable to the `val_400_0.50_90.ffcv` file as follows:

```ini
...
[dataloader]
train_dataset=<path/to/train_400_0.50_90.ffcv>
val_dataset=<path/to/val_400_0.50_90.ffcv>
...
```

Next, obtain your wandb API key so the code will have access to your wandb workspace:
1. Go to wandb.ai and login to your account.
2. From the top-right corner of the screen, select your profile icon and then go to Settings. 
3. From the User Settings page, copy the API key. This API key is needed to log the experiments to wandb. We save the model weights, loss/accuracy plots, training configurations, and code files to wandb.ai for easy access in the future.

Now we can begin training models randomly sampled from the search space: 

```bash
torchrun --nnodes=1 --nproc_per_node=4 train_nmodels_proxified.py --cfg_path configs/conf_local.cfg --num_models 850 --seed <seed> --wandb-api-key <API key here> --wandb-project <wandb project name> --wandb-entity <wandb username here>
```

The script will train 850 models on a node with 4 GPUs. Multiple instances of the script can be run in parallel using multiple nodes and running the above command with a different `seed` argument to parallelize the dataset collection. When the above command is executed, a new wandb.ai run will be created in your wandb workspace. The training progress can be observed in the training log and also on wandb.ai. 

###### Note: Due to node downtimes/maintenance/memory-overflows, we had to stop training multiple times and resume the dataset collection with a different seed each time. This does not impact the integrity of dataset collection since the architectures are sampled randomly. 

After the training has finished, keep a note of the job-id which is assigned to your wandb run as postfix to the name of the run. This job ID, along with wandb user info can be used to obtain the dataset from wandb. Script <TODO> collects the dataset from wandb and generates a csv file containing the architectures, their FLOPs, # of parameters, accuracy etc.

###### Caveat: The FFCV dataloader is slow in the first epoch since it loads the dataset to the memory. Please be patient.

## Collection of throughput/latency on accelerators
#### FPGAs
This requires some familiarity with [Vitis AI](https://github.com/Xilinx/Vitis-AI/tree/2.5) and FPGA design flow. Please make sure Vitis AI and `vai_q_pytorch` are installed on the host machine. `vai_q_pytorch` is an 8-bit quantizer that comes bundled with Vitis AI inside the provided docker containers, so installing Vitis-AI also installs the quantizer. We utilized [Vitis AI 2.5](https://github.com/Xilinx/Vitis-AI/tree/2.5) since version 3.0 was released only recently, however the version update does not have any impact on the performance measurements we made. We used two FPGAs 1) [Zynq UltraScale+ MPSoC ZCU102](https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html) and 2) [Versal AI Core Series VCK190](https://www.xilinx.com/products/boards-and-kits/vck190.html). The corresponding DPUs are [DPUCZDX8G_ISA1_B4096](https://docs.xilinx.com/r/en-US/pg338-dpu?tocId=~72l0MosWV8p9MbkDlnw8Q) for ZCU102 and [DPUCVDX8G_ISA3_C32B3](https://docs.xilinx.com/r/en-US/pg389-dpucvdx8g/Introduction) for VCK190. Other FPGA devices such as VCK5000 and ZCU104 are also supported, while some server FPGAs such as Alveo U50/U250/U280 are not supported since their DPUs currently do not support the `hard sigmoid` operation utilized by the squeeze-excite operation 
######  (Perhaps these 'unsupported' FPGAs can still be used but by sampling only the architectures from the search space that do not utilize Squeeze-Excitation layers in any of the blocks. This however considerably reduces the size of search space to about $10^9$ architectures only. The search results on the supported FPGAs using multi-objective optimization on accuracy+throughput show that FPGA-targeted models do not prefer the squeeze-excitation anyways owing to its memory-bound nature).

###### Also please note that FPGA code in the dir `fpga_code` requires Vitis-AI installed on the host machine and FPGA setup with the petalinux. Please feel free to file an issue if you need some help with these steps.

We offer the scripts for: 1) Quantization and compilation for the two FPGAs, and 2) Running compiled models on the FPGAs. 

###### The code can easily be adopted to other supported FPGAs by changing the `TARGET` and `ARCH` variables in the `compile.sh` script. Only the compile phase is different when targeting different FPGAs; quantization is FPGA-device agnostic, and the same runners are utilized to execute on both FPGAs.

##### Model Quantization and *.xmodel Compilation
This is performed on the host machine inside the `vitis-ai-pytorch` conda environment that comes with the Vitis AI docker container. We recommend the docker GPU container for quantization since this requires model evaluation, however a weaker GPU will do since no training is involved. The subsequent instructions assume Vitis-AI is installed and you are inside the `vitis-ai-pytorch` conda env inside the docker container. Test if the quantization and compilation packages are correctly installed by running `python3 -c "from pytorch_nndct.apis import torch_quantizer, dump_xmodel"` inside the conda env.

The script takes the wandb job ids as input and generates quantized and compiled models in `dataset_log` directory. The compiled models are located in the `compiled_models` subdirectory with the *.xmodel extension. These models can be deployed on the target FPGAs. Transfer the compiled xmodel files to the FPGA using `scp` or some other mechanism. FPGAs should be running petalinux according the device setup (e.g., [here](https://github.com/Xilinx/Vitis-AI/tree/2.5/setup/mpsoc) for ZCU102 and [here](https://github.com/Xilinx/Vitis-AI/tree/2.5/setup/vck190) for VCK190) and network interfaces can be activated by using an ethernet connection with the FPGA.

##### Execute on target FPGAs
This is performed on the target FPGA running petalinux. After the xmodel files have been transferred to the target FPGA, place the files in `models_to_eval` directory. Also please copy a small subset of the ImageNet dataset to `gen_images` directory. We utilized a subset of around 1000 images for throughput measurements as using more images than this does not have a significant impact on the measurements. The script measures and stores model names and the latency/throughputs into the `results/eval.txt` file. We measure latency using single threaded execution with batch size of 1 on the ZCU102 board and batch size of 6 on the VCK190 board. VCK190 does not support lower than batch size of 6. Throughput is measured by maximizing the performance on a small random pool of models; Thread count of 5 and 6 give the highest throughput on ZCU102 and VCK190, respectively. 

---
#### TPUs
Inference throughput is reported using TPUv2 and TPUv3s. We utilize GCP with `tpu-vm-pt-1.13` TPU VMs for all our experiments. Throughput is measured at batch size of 64 and 128 on TPUv2 and v3, respectively. This is because higher batch sizes did not yield a noticeable improvement in throughput. 

The script `measure_throughput_tpu.py` handles inference throughput measurements on the TPU devices. We utilize a single TPU core for these measurements instead of all 8 cores. This is because single core measurements of throughput are a lot more stable than multi-core. We also measure the error in measurements by repeating the measurements and computing the mean and standard deviation of throughput of the measurements across multiple runs on the same model.

###### Caveat: TPUs have a long warmup. We measure the inference throughput on the training! dataset since the TPUs spend more than 50k samples in warming up, leading to inaccurate measurements if only the validation dataset is used. Using the ~1.2M samples of the training set, we are able to ensure that the TPUs reach a steady state before measuring the mean and standard deviation of throughput. 

###### Future Plans: Add a surrogate for training throughput as some NAS works consider training speed as an optimization objective (e.g., EfficientNetsV2). This is however expensive to collect since the first 1-2 training epochs are slow/warmup due to XLA graph compilation and caching, and will need to be discarded from measurements. Estimated compute cost of several weeks using 4-5 TPUs is expected.

---
#### GPUs
The script `measure_throughput_gpu.py` handles inference throughput measurements on the GPU devices. Similar to the TPU measurements, we avoid the GPU warmup by discarding an evaluation run through the validation set. We then measure throughput on the entire validation set twice, logging the mean and standard deviation of the measurements. The measurements are a lot more stable compared to the TPU measurements, hence using the validation set for throughput measurements results in sufficiently accurate measurements. We perform measurements on an RTX 3090 and an A100 40GB GPU. Please see `trainval\trainval_exact_gpu.py` file and `measure_throughput_gpu` function for the details of measurements.

###### Caveat: Only the mean measurements for throughput/latency are utilized to train the surrogates. The standard deviation measurements are for internal testing only. Perhaps these error values can somehow be utilized to allow the surrogates to output the uncertainty in predictions, rather than using the ensemble prediction uncertainty as the noise in throughput.

---

### Update 25 Aug 2023

Added code for ECG, Satellite, and DeepSEA datasets. Please see corresponding folders above. 

The training/evalation of these additional datasets follows the same procedure as that highlighted for ImageNet dataset above.
