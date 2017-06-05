

import numpy as np
import glob

import os
import time

import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS
data_dim = 1024
nb_classes = 4716


if __name__ == '__main__':
  flags.DEFINE_string("train_dir", "/home/ksh/youtube_8m/models/",
                      "The directory to load the model files from.")
  flags.DEFINE_string("output_file", "/home/ksh/youtube_8m/",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "/home/ksh/data/yt8m/train*",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 16,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.
  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.
  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.
  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, label_batch, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return video_id_batch, video_batch, label_batch, num_frames_batch

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    video_id_batch, video_batch, label_batch, num_frames_batch = get_input_data_tensors(reader, data_pattern, batch_size)
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=False)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)

    graph = tf.get_default_graph()
    feature1 = [graph.get_tensor_by_name("tower/fully_connected_2/Relu6:0")]
    feature2 = [graph.get_tensor_by_name("tower/fully_connected_3/Sigmoid:0")]
    for i in range(1,8):
      feature1.append(graph.get_tensor_by_name("tower_"+str(i)+"/fully_connected_2/Relu6:0"))
      feature2.append(graph.get_tensor_by_name("tower_"+str(i)+"/fully_connected_3/Sigmoid:0"))
    labels = tf.get_collection("labels")[0]
    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    file_num = 0
    start_time = time.time()


    try:
      while not coord.should_stop():
        video_id_batch_val_tot = []
        feature1_val_tot = []
        feature2_val_tot = []
        labels_val_tot = []
        for i in range(1024):
          video_id_batch_val, video_batch_val,num_frames_batch_val, labels_val = sess.run([video_id_batch, video_batch, num_frames_batch, label_batch])
          feature1_1_val,feature1_2_val,feature1_3_val,feature1_4_val,feature1_5_val,feature1_6_val,feature1_7_val,feature1_8_val,feature2_1_val,feature2_2_val,feature2_3_val,feature2_4_val,feature2_5_val,feature2_6_val,feature2_7_val,feature2_8_val, = sess.run(feature1+feature2, feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val})
          feature1_val_tot.append(feature1_1_val)
          feature1_val_tot.append(feature1_2_val)
          feature1_val_tot.append(feature1_3_val)
          feature1_val_tot.append(feature1_4_val)
          feature1_val_tot.append(feature1_5_val)
          feature1_val_tot.append(feature1_6_val)
          feature1_val_tot.append(feature1_7_val)
          feature1_val_tot.append(feature1_8_val)
          feature2_val_tot.append(feature2_1_val)
          feature2_val_tot.append(feature2_2_val)
          feature2_val_tot.append(feature2_3_val)
          feature2_val_tot.append(feature2_4_val)
          feature2_val_tot.append(feature2_5_val)
          feature2_val_tot.append(feature2_6_val)
          feature2_val_tot.append(feature2_7_val)
          feature2_val_tot.append(feature2_8_val)
          video_id_batch_val_tot.append(video_id_batch_val)
          labels_val_tot.append(labels_val)

          now = time.time()
          num_examples_processed += len(video_batch_val)
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
        logging.info(file_num)
        feature1_val_tot = np.concatenate(feature1_val_tot, axis=0)
        feature2_val_tot = np.concatenate(feature2_val_tot, axis=0)
        video_id_batch_val_tot = np.concatenate(video_id_batch_val_tot, axis=0)
        labels_val_tot = np.concatenate(labels_val_tot, axis=0)
        logging.info(video_id_batch_val_tot.shape)
        logging.info(labels_val_tot.shape)
        with gfile.Open(out_file_location + 'id/' + 'dobf' + '_id' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, video_id_batch_val_tot)
        with gfile.Open(out_file_location + 'feature/' + 'dobf' + '_feature1_' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, feature1_val_tot)
        with gfile.Open(out_file_location + 'feature/' + 'dobf' + '_feature2_' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, feature2_val_tot)
        with gfile.Open(out_file_location + 'label/' + 'dobf' + '_label' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, labels_val_tot)
        logging.info(file_num)
        file_num+=1



    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ')
        logging.info(file_num)
        feature1_val_tot = np.concatenate(feature1_val_tot, axis=0)
        feature2_val_tot = np.concatenate(feature2_val_tot, axis=0)
        video_id_batch_val_tot = np.concatenate(video_id_batch_val_tot, axis=0)
        labels_val_tot = np.concatenate(labels_val_tot, axis=0)
        logging.info(feature1_val_tot.shape)
        logging.info(video_id_batch_val_tot.shape)
        logging.info(labels_val_tot.shape)
        with gfile.Open(out_file_location + 'id/' + 'dobf' + '_id' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, video_id_batch_val)
        with gfile.Open(out_file_location + 'feature/' + 'dobf' + '_feature1_' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, feature1_val_tot)
        with gfile.Open(out_file_location + 'feature/' + 'dobf' + '_feature2_' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, feature2_val_tot)
        with gfile.Open(out_file_location + 'label/' + 'dobf' + '_label' + str(file_num)+'.npy', "w+") as out_file:
          np.save(out_file, labels_val)
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
    FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
