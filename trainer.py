# Copyright 2020 xueyuanl. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import six
import tensorflow as tf
import tensorflow_hub as hub
import yaml
from absl import logging

import trainer_lib as lib

_DEFAULT_HPARAMS = lib.get_default_hparams()


def _check_keras_dependencies():
    """Checks dependencies of tf.keras.preprocessing.image are present.
    This function may come to depend on flag values that determine the kind
    of preprocessing being done.
    Raises:
      ImportError: If dependencies are missing.
    """
    try:
        tf.keras.preprocessing.image.load_img(six.BytesIO())
    except ImportError:
        print("\n*** Unsatisfied dependencies of keras_preprocessing.image. ***\n"
              "To install them, use your system's equivalent of\n"
              "pip install tensorflow_hub[make_image_classifier]\n")
        raise
    except Exception as e:  # pylint: disable=broad-except
        # Loading from dummy content as above is expected to fail in other ways.
        pass


def _assert_accuracy(train_result, assert_accuracy_at_least):
    # Fun fact: With TF1 behavior, the key was called "val_acc".
    val_accuracy = train_result.history["val_accuracy"][-1]
    accuracy_message = "found {:f}, expected at least {:f}".format(
        val_accuracy, assert_accuracy_at_least)
    if val_accuracy >= assert_accuracy_at_least:
        print("ACCURACY PASSED:", accuracy_message)
    else:
        raise AssertionError("ACCURACY FAILED:", accuracy_message)


def _set_gpu_memory_growth():
    # Original code reference found here:
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("All GPUs will scale memory steadily")
    else:
        print("No GPUs found for set_memory_growth")


def main(image_dir,
         tfhub_module='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
         image_size=224,
         saved_model_dir='trained_model/new_model',
         tflite_output_file='trained_model/new_mobile_model.tflite',
         labels_output_file='trained_model/class_labels.txt',
         summaries_dir=None,
         assert_accuracy_at_least=None,
         train_epochs=_DEFAULT_HPARAMS.train_epochs,
         do_fine_tuning=_DEFAULT_HPARAMS.do_fine_tuning,
         batch_size=_DEFAULT_HPARAMS.batch_size,
         learning_rate=_DEFAULT_HPARAMS.learning_rate,
         momentum=_DEFAULT_HPARAMS.momentum,
         dropout_rate=_DEFAULT_HPARAMS.dropout_rate,
         set_memory_growth=False):
    """Main function to be called by absl.app.run() after flag parsing."""

    # del args
    def _get_hparams_from_flags():
        """Creates dict of hyperparameters from flags."""
        return lib.HParams(
            train_epochs=train_epochs,
            do_fine_tuning=do_fine_tuning,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            dropout_rate=dropout_rate)

    _check_keras_dependencies()
    hparams = _get_hparams_from_flags()

    image_dir = image_dir or lib.get_default_image_dir()

    if set_memory_growth:
        _set_gpu_memory_growth()

    model, labels, train_result = lib.make_image_classifier(
        tfhub_module, image_dir, hparams, image_size, summaries_dir)
    if assert_accuracy_at_least:
        _assert_accuracy(train_result, assert_accuracy_at_least)
    print("Done with training.")

    if labels_output_file:
        with tf.io.gfile.GFile(labels_output_file, "w") as f:
            f.write("\n".join(labels + ("",)))
        print("Labels written to", labels_output_file)

    saved_model_dir = saved_model_dir
    if tflite_output_file and not saved_model_dir:
        # We need a SavedModel for conversion, even if the user did not request it.
        saved_model_dir = tempfile.mkdtemp()
    if saved_model_dir:
        tf.saved_model.save(model, saved_model_dir)
        print("SavedModel model exported to", saved_model_dir)

    if tflite_output_file:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        lite_model_content = converter.convert()
        with tf.io.gfile.GFile(tflite_output_file, "wb") as f:
            f.write(lite_model_content)
        print("TFLite model exported to", tflite_output_file)


def _ensure_tf2():
    """Ensure running with TensorFlow 2 behavior.
    This function is safe to call even before flags have been parsed.
    Raises:
      ImportError: If tensorflow is too old for proper TF2 behavior.
    """
    logging.info("Running with tensorflow %s and hub %s",
                 tf.__version__, hub.__version__)
    if not tf.executing_eagerly():
        raise ImportError("Sorry, this program needs TensorFlow 2.")


def train_images():
    """Entry point equivalent to executing this file."""
    with open('conf.yaml') as f:
        train_image_dir = yaml.safe_load(f)['train_image_dir']

    _ensure_tf2()
    main(train_image_dir)


if __name__ == "__main__":
    train_images()
