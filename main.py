import argparse
import os
from tqdm import tqdm

from PIL import Image
import cv2
import numpy as np
from preprocess_image import preprocess_image
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--watermark_type', default='istock', type=str,
                    help='The watermark type')
parser.add_argument('--checkpoint_dir', default='model/', type=str,
                    help='The directory of tensorflow checkpoint.')

#checkpoint_dir = 'model/'


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()

    # Check if the input is a directory
    if os.path.isdir(args.image):
        input_dir = args.image
        output_dir = args.output

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the list of image files
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Create a tqdm progress bar for the images
        for filename in tqdm(image_files, desc="Processing Images"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            image = Image.open(input_path)
            input_image = preprocess_image(image, args.watermark_type)


            # Create a new TensorFlow graph
            tf.compat.v1.reset_default_graph()

            with tf.compat.v1.Session() as sess:
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = model.build_server_graph(FLAGS, input_image)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)

                # Load pretrained model
                vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.compat.v1.train.load_variable(args.checkpoint_dir, from_name)
                    assign_ops.append(tf.compat.v1.assign(var, var_value))
                sess.run(assign_ops)

                print(f'\nProcessing image: {filename}')
                result = sess.run(output)
                cv2.imwrite(output_path, cv2.cvtColor(result[0][:, :, ::-1], cv2.COLOR_BGR2RGB))
                print(f'Image saved to: {output_path}')

    else:
        # Process a single image as before
        image = Image.open(args.image)
        input_image = preprocess_image(image, args.watermark_type)

        # Create a new TensorFlow graph
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            # Load pretrained model
            vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.compat.v1.train.load_variable(args.checkpoint_dir, from_name)
                assign_ops.append(tf.compat.v1.assign(var, var_value))
            sess.run(assign_ops)

            print('Model loaded.')
            result = sess.run(output)
            cv2.imwrite(args.output, cv2.cvtColor(result[0][:, :, ::-1], cv2.COLOR_BGR2RGB))
            print(f'Image saved to: {args.output}')