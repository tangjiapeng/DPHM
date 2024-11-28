import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
from PIL import Image
import cv2
import imageio
from natsort import natsorted

current_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
                        default=os.path.join(current_dir, 'animation'))
    parser.add_argument('-f', '--filename', type=str,
                        default='rotation')
    parser.add_argument('-ext', '--extension', type=str, default='jpg') #png
    #parser.add_argument('-len', '--mesh-len', type=int, default=100)
    parser.add_argument('-fps', '--fps', type=int, default=15)
    parser.add_argument('-dr', '--downsample_ratio', type=int, default=1)
    ####
    parser.add_argument('-sid', '--start_idx', type=int, default=0)
    parser.add_argument('-eid', '--end_idx', type=int, default=0)
    parser.add_argument('-ng', '--no_gif', action='store_true')
    ####
    parser.add_argument('-xmin', '--x_min', type=int, default=0) # 560
    parser.add_argument('-ymin', '--y_min', type=int, default=0) # 80
    parser.add_argument('-bsx', '--box_size_x', type=int, default=800) # 800
    parser.add_argument('-bsy', '--box_size_y', type=int, default=800) # 800
    parser.add_argument('-wt', '--white', action='store_true')
    args = parser.parse_args()

    fns = natsorted(os.listdir(args.filename_input))
    fns = [f for f in fns if f.endswith(args.extension)]
    
    if args.white:
        white_image_dir = args.filename_input[:-1] + '_white'
        os.makedirs(white_image_dir, exist_ok=True)

    # draw object from different view
    image_bgr_list = []
    image_rgb_list = []
    
    if args.end_idx == 0:
        args.end_idx = len(fns)
    for num in range(args.start_idx, args.end_idx):
        #print(num)
        image_path = os.path.join( args.filename_input, '{:s}'.format(fns[num]) )
        img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #img_bgr = cv2.imread(image_path)
        #print(img_bgr.shape)

        if img_bgr.shape[-1] == 4:
            mask = (img_bgr[:, :, 3] == 0)
            img_bgr[mask, 0:3] = 255
            img_bgr = img_bgr[:, :, :3]

            #mask = (img_bgr.sum(axis=2) < 10)
            #img_bgr[mask, :] = np.ones(3, dtype='uint8') * 255 
        
        if args.x_min >0  or args.y_min >0:
            img_bgr = img_bgr[args.y_min+args.y_min:args.box_size_y, args.x_min: args.x_min+args.box_size_x, :]
        
        if args.white:
            image_path_white = os.path.join(white_image_dir, '{:05d}.{:s}'.format(num, args.extension))
            cv2.imwrite(image_path_white, img_bgr)


        height, width, layers = img_bgr.shape
        size = (width, height)
        size_dr = (int(width//args.downsample_ratio), int(height//args.downsample_ratio))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_bgr_list.append(cv2.resize(img_bgr, size_dr, interpolation = cv2.INTER_AREA))
        image_rgb_list.append(cv2.resize(img_rgb, size_dr, interpolation = cv2.INTER_AREA))

    # out = cv2.VideoWriter(os.path.join(args.filename_input, '%s_fps%d_dr%d.mp4'%(args.filename, args.fps, args.downsample_ratio)), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), args.fps, size)
    # for num in range(args.end_idx-args.start_idx):
    #     out.write(image_bgr_list[num])
    # out.release()
    
    writer = imageio.get_writer(os.path.join(args.filename_input, '%s_fps%d_dr%d.mp4'%(args.filename, args.fps, args.downsample_ratio)), fps=args.fps)
    for im in image_rgb_list:
        writer.append_data(im)
    writer.close()

    if not args.no_gif:
        imageio.mimsave(os.path.join(args.filename_input, '%s_fps%d_dr%d.gif'%(args.filename, args.fps, args.downsample_ratio)), image_rgb_list, "GIF")
        

if __name__ == '__main__':
    main()
