# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_integer(
    "hidden_size", 4096,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")



class PartDnnModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   hidden_size=4096,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    hid_1_activations=[]
    hid_2_activations=[]
    hid_3_activations=[]
    predictions=[]
    init_predictions=[]
    num_output=[0,1500,3000,4716]
    for i in range(len(num_output)-1):
      hid_1_activations.append(slim.fully_connected(
          model_input,
          hidden_size,
          activation_fn=tf.nn.relu6,
          biases_initializer=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty)))

      hid_2_activations.append(slim.fully_connected(
          hid_1_activations[i],
          hidden_size,
          activation_fn=tf.nn.relu6,
          biases_initializer=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty)))
      hid_3_activations.append(slim.fully_connected(
          hid_2_activations[i],
          hidden_size,
          activation_fn=tf.nn.relu6,
          biases_initializer=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty)))

      init_predictions.append(slim.fully_connected(
          hid_3_activations[i],
          vocab_size,
          activation_fn=tf.nn.sigmoid,
          biases_initializer=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty)))
      predictions.append(init_predictions[i][:,num_output[i]:num_output[i+1]])


    init_probabilities = tf.stack(init_predictions,2)
    final_probabilities = tf.concat(predictions,1)


    return {"predictions": final_probabilities,"init_predictions": init_probabilities}




class DnnModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   hidden_size=4096,
                   **unused_params):

        
    hidden_size = hidden_size or FLAGS.hidden_size

    hid_1_activations = slim.fully_connected(
        model_input,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hid_2_activations = slim.fully_connected(
        hid_1_activations,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hid_3_activations = slim.fully_connected(
        hid_2_activations,
        hidden_size,
        activation_fn=tf.nn.relu6,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    predictions = slim.fully_connected(
        hid_3_activations,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        biases_initializer=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(l2_penalty))

    return {"predictions": predictions}





class DnnModel2(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   hidden_size=4096,
                   **unused_params):

        
    hidden_size = hidden_size or FLAGS.hidden_size

    hid_1_activations = slim.fully_connected(
        model_input,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hid_2_activations = slim.fully_connected(
        hid_1_activations,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hid_3_activations = slim.fully_connected(
        hid_2_activations,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hid_4_activations = slim.fully_connected(
        hid_3_activations,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    hid_5_activations = slim.fully_connected(
        hid_4_activations,
        hidden_size,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    predictions = slim.fully_connected(
        hid_5_activations,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))

    return {"predictions": predictions}





class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
