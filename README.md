# Stitching

Notebooks and code to perform experiments on model-stitching:

- **exp1_ms_with_random_dataset**
    - *exp1f_weak_ResNet-18_synthdata_rank.ipynb* (Optionally) trains ResNet-18 models to use as receivers, and stitches ‘synth’ data into them as a
sender. Uses https://github.com/DHLSmith/jons-tunnel-effect.git packages to record rank and accuracy. Change seed and rerun to generate differently initialised and stitched
models.
    - *exp1g_VGG19_synthdata_rank.ipynb* (Optionally) trains VGG19 models as receivers and stitches ‘synth’ data into them as a sender. Records rank and accuracy. Rerun with new seed to generate more models.
	- *exp1m_resnet50_imagenet_synthdata.ipynb* stitches 'synth' data into a ResNet-50 pretrained on ImageNet
	- *exp1n_weak_ResNet-18_synthdata_morenoise.ipynb* Using stitches 'synth' data into ResNet-18 models at one layer, for different radiuses of synthetic noise cluster. 
- **exp2_ms_with_colorMNIST** 
    - *exp2e_colorMNIST_weak_bias_data_unbias_rcver_rank.ipynb* (Optionally) Trains various models and stitches them as senders into an ‘unbias’-trained receiver. Records rank and accuracy.
    - *exp2e_b_colorMNIST_weak_senders_own_data_unbias_rcver_rank.ipynb* As with exp2e, but the stitch is trained and measured using the dataloader for the sender model rather than the biased data loader
    - *exp2e_c_colorMNIST_weak_senders_own_data_rank.ipynb* stitch BW to Colour-Only and vice versa, but the stitch is trained and measured using the dataloader for the sender model
    - *exp2e_d_colorMNIST_weak_self-stitch_bw_bgonly.ipynb* stitch BW to itself and Colour-Only to itself. The stitch is trained and measured using the dataloader for the sender model
	- *exp2g_a_colorMNIST_weak_self-stitch_bw_bgonly_classremap.ipynb* self-stitches BW  (Greyscale-MNIST) to itself, but with a class offset (e.g. mapping digits of 0 to 3). Also self-stitches BGOnly (Colour-Only) with a class offset (e.g. Red to Green)
	- *exp2h_colorMNIST_VGG19_bg_bgonly.ipynb* used to stitch specific VGG19 models because some initialisations had stitched badly
    - *exp2j_colorMNIST_VGG19_bias_data_unbias_rcvr.ipynb* stitch Various VGG19 models to Digit model using Correlated dataloader
    - *exp2j_a_colorMNIST_VGG19_unbias_data_randinit_sender_unbias_rcvr.ipynb* stitch a randomly initialised VGG19 network to an Digit network using Digit data
	- *exp2j_b_colorMNIST_VGG19_bias_data_randinit_sender_unbias_rcvr.ipynb* Stitching a randomly initialised model as sender into VGG19 receiver to compare performance against trained sender models.
    - *exp2k_bias_data_bg_colour_sender_unbias_receiver.ipynb* stitch Colour to Digit using Correlated DataLoader
    - *exp2l_bias_data_bg_colour_sender_unbias_receiver_l1_reg.ipynb* Used for evaluating regularisation strength
    - *exp2m_bias_data_randinit_sender_unbias_receiver.ipynb* stitch randomly initialised model (ResNet-18) to Digit using Correlated Dataloader
    - *exp2n_bw_data_randinit_sender_bgonly_receiver.ipynb* stitch randomly initialised ResNet-18 to Colour-Only receiver using BW Dataloader
	- *exp2n_b_bgonly_data_randinit_sender_bw_receiver.ipynb* stitch randomly initialised model (ResNet-18) to BW (Greyscale-MNIST) receiver with BG-Only Dataloader
    - *plot_images_for_report.ipynb* Creates and saves sample images for use in reports.
- **exp4_ms_unbias_to_various_with_colourMNIST**
    - *exp4a_bias_data_unbias_sender.ipynb* (Optionally) Trains various models and stitches an ‘unbias’-trained sender into them as receivers. Records rank and accuracy.
- **ReferenceCode**
    - *colour_mnist.py* used to create variant data generators. Note that this code is covered by an MIT License
    - *stitch_utils.py* Package containing functions used by the experiment notebooks.
- *conda_env_file.yml* YAML file to recreate conda environment used for the experiments. To set up equivalent environment use:
`$ conda env create --name tempenv --file ./conda_env_file.yml`
- *plot_ranks_and_accuracy.ipynb* notebook based on rank.ipynb from ttps://github.com/DHLSmith/jons-tunnel-effect.git
- *plot_ranks_and_accuracy_for_randinit.ipynb* For plotting graphs from the random initialisation experiments

## Naming of datasets and models
The code was originally created for an MSc Dissertation, but for the NeurIPS workshop paper the datasets were renamed when plotting graphs for clarity. 

| MSc    | NeurIPS     | Description |
| ------ | ------      | ------ |
| unbias | Digit       | Images of digits on colour backgrounds where the colour is unrelated to the class of digit |
| bias   | Correlated  | Images of digits with colour backgrounds where the colour correlates with the class of digit |
| bg     | Colour      | As for unbias/Digit, but the target is the class of colour rather than the digit |
| bgonly | Colour-Only | As for bg/Colour, but there is no digit: only colour |
| bw     | BW          | Black & White original MNIST images. (also Greyscale-MNIST)
| synth  | Noise       | Synthetically generated data in representation space. Classes are randomly chosen points and samples within each class are noise around those points |

# Trained Models
Models trained early on in the research project were saved as (for example) `./results_4_epochs/2024-08-06_12-57-58_SEED60_EPOCHS4_BGN0.1_exp2e_ResNet-18_bw_mnist.weights`
In general, stitch accuracy and rank measurements were later run on the same set of models to reduce processing time and increase consistency. 
The stitch experiments only worked on pairs of models in the same training run - we did not cross compare due to time constrainst. 
VGG19 models are not included in this repository due to their size (ca 500MB)

# License
Most of this project is made available under the BSD license (see ./LICENSE). 
However ./ReferenceCode/colour_mnist.py is derived from https://github.com/DHLSmith/clovaai-rebias/blob/master/datasets/colour_mnist.py and is covered by the MIT License.
