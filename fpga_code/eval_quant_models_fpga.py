import os
from run_on_fpga import execute_on_fpga


def main():
    models_dir = './models_to_eval/'
    image_dir = './gen_images/images/'
    threads = 4
    dataloader = 'opencv'
    models_list = sorted(os.listdir(models_dir))
    print(models_list)
    for model in models_list:
        model_path = os.path.join(models_dir, model)
        fps, accuracy = execute_on_fpga(image_dir, threads, model_path, dataloader)
        print(f'FPS: {fps}, Accuracy: {accuracy}')

if __name__ == '__main__':
    main()

