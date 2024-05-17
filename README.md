# high-risk-Cervical-Precancerous-Screening

## low resolution WSI format
A low resolution WSI is acquired by redundantly imaging different region of a sample. Our system supports any number of images for a slide.
In demo, the raw images are 3840 * 2160 pixels (0.87 um/pixel) which will be further center-cropped to 1600 * 1600 pixels. All the images should be
saved in a dir.


## file
+ core -> define models
+ utils-> to read wsi
+ weights -> the model weights, down load at [weights](https://huggingface.co/BruceAwake/high-risk-Cervical-Precancerous-Screening/tree/main/weigths)
  + stage_one_weight_file
  + stage_two_weight_file
  + down load at 
+ engine.py -> to inference
+ visualization.py -> to show the results 
+ WSI_DEMO -> a lrwsi demo file, down load at [demo file](https://huggingface.co/BruceAwake/high-risk-Cervical-Precancerous-Screening/tree/main/WSI_DEMO) 
  + image of a view 1.bmp
  + image of a view 2.bmp
+ main.py -> a demo script


## Quick start
### software and hard ware:
> we have test the project on:
> + operation system of the server is Linux version 5.4.0-150-generic. 
> + server equipped with four NVIDIA TITAN V graphics processing unit (UPG)
> + Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz. 
> + CUDA Version is 11.4.     

In fact, the hardware and software requirements for this project are minimal, you can set smaller batch_size to meet the need which just will cost more time.

### step 0
> down load weight file, and save it in ./weights  (essential)
> down demo lrwsi, and sace it in ./WSI_DEMO
### step 1
> create conda env, and make sure libs in requirements.txt is satisfied.  
> ```shlll
> conda create -n your_env_name
> ```
> activate the env by:  
> ```shell
> conda activate your_env_name
> ```

### step 2
> make sure your data is organized as (like dir WSI_DEMO)  
> A_LOW_RESOLUTION_WSI_DIR_PATH/  
> &nbsp;&nbsp;&nbsp;&nbsp;|-- image1.jpg  
> &nbsp;&nbsp;&nbsp;&nbsp;|-- iamge2.jpg  
> &nbsp;&nbsp;&nbsp;&nbsp;......  
### step 3
> run following command (demo):  
> ```shell
> python main.py --batch_size 128 --save_dir demo_result --wsi_dir WSI_DEMO --show_mode top
> ```
> --batch_size: the batch size for the first stage model    
> --save_dir: dir to save results  
> --wsi_dir: lrwsi dir to test
> --show_mode: if [top], the results are the top 10 instance with heat map mask  
> ---
> you can also set --show_mode [view], and run following command (demo):  
> ```shell
> python main.py --batch_size 128 --save_dir demo_result --wsi_dir WSI_DEMO --show_mode view --view_full_path WSI_DEMO\202101141632580407.bmp
> ```
> --view_full_path: if --show_mode is [view], the full path to a certain image of a view in a slide should be added    
> --show mode: if --show_mode is [view], the results are the top 200 instance in the image of a view

## To support other format wsi
> to make use of our method on your own low resolution wsi data, you just need to modify the script: utils/smallwsiread.py, where:
> + replace the function [block2instances] defined in [utils.SmallWsiRead.read] as you like
> + make sure [self.instances] is like: {'filename_corW_corH': numpy_format_im, }
>   + corW, cor_H: crop position relative to the top left corner of the image
>   + numpy_format_im: (256, 256, 3) \ RGB \ 0-255 \ np.uint8
> + make sure there is more than 200 insances in a slide

