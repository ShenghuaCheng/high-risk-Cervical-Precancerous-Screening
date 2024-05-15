# -*- coding: utf-8 -*-
import os
from engine import infer_slide
from visulization import see_a_view
from visulization import see_stage_two_top_10
import datetime
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization the results')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='./demo_result', help='results save dir')
    parser.add_argument('--wsi_dir', type=str, default='wsi_demo', help='low resolution wsi dir')
    parser.add_argument(
        '--show_mode', type=str, default='top',
        help='choose [view] to see result of a view if there is any top instance in it, or [top] to see top 10 instances in a slide')
    parser.add_argument(
        '--view_full_path', type=str,
        help='the full path if image of view in a slide when show_mode:[view]'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        print('make dir: {}'.format(args.save_dir))

    print('start to infer the slide')
    ss = datetime.datetime.now()
    infer_slide(args.wsi_dir, args.save_dir, 128)
    ee = datetime.datetime.now()
    print('time cost: {} s'.format((ee-ss).seconds))

    if args.show_mode == 'top':
        print('---------------- show top 10 instance -----------------------------------------')
        format_ = os.listdir(args.wsi_dir)[0].split('.')[-1]
        see_stage_two_top_10(args.save_dir, args.wsi_dir, format_, args.save_dir)

    elif args.show_mode == 'view':
        print('---------------- show top 200 instance in a image of view ---------------------')
        if args.view_full_path:
            see_a_view(args.save_dir, args.view_full_path, args.save_dir)
        else:
            print('please set --view_full_path the full path of the image of view in the slide')

    else:
        print('please set --show_mode as [top] or [view]')

