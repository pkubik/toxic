import tensorflow as tf

from toxic.data.input import EMBEDDING_SIZE
from toxic.data.utils import CLASSES
from toxic.utils import DictWrapper


class Features(DictWrapper):
    def __init__(self):
        self.id = None
        self.text = None
        self.text_length = None


class Labels(DictWrapper):
    def __init__(self):
        self.classes = None


class Params(DictWrapper):
    def __init__(self):
        self.num_epochs = 80
        self.batch_size = 64
        self.max_word_idx = None
        self.learning_rate = 0.001
        self.regularization_scale = 0.0001


NUM_CLASSES = sum(1 for _ in CLASSES)

DEFAULT_PARAMS = Params().as_dict()


def model_fn(mode, features, labels, params):
    _params = Params.from_dict(params)
    _features = Features.from_dict(features)

    _labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        _labels = Labels.from_dict(labels)

    return build_model(mode, _features, _labels, _params)


def build_model(mode: tf.estimator.ModeKeys,
                features: Features,
                labels: Labels,
                params: Params) -> tf.estimator.EstimatorSpec:
    """
    Builds prediction model

    Tuples (X, Y, Z) in the comments denote dimension meanings where:
    - B is batch dimension
    - L is tokens dimension
    - E is original word embedding dimension
    - C is output classes dimension
    - other symbols stand for arbitrary local dimension (e.g. in hidden states)

    :param mode: whether training, testing or predicting (affects dropout, batchnorm, etc.)
    :param features: inputs available in all modes
    :param labels: inputs available only in the training and testing mode
    :param params: hyperparameters
    :return: `tf.estimator.EstimatorSpec` required for the rest of the pipeline
    """
    with tf.variable_scope('features'):
        features.id = tf.placeholder_with_default(features.id, [None], 'id')  # (B)
        features.text = tf.placeholder_with_default(features.text, [None, None], 'text')  # (B, L)
        features.text_length = tf.placeholder_with_default(features.text_length, [None], 'text_length')  # (B)

    global_step = tf.train.get_global_step()

    with tf.device("/cpu:0"):
        embeddings = tf.placeholder(tf.float32, [None, EMBEDDING_SIZE], name='embeddings')  # (V, E); V - vocabulary
        embedded_text = tf.nn.embedding_lookup(embeddings, tf.nn.relu(features.text))  # (B, L) -> (B, L, E)

    with tf.variable_scope("encoder"):
        final_embedding = tf.reduce_sum(embedded_text, -2)  # (B, L, E) -> (B, E)

    with tf.variable_scope("output"):
        final_layer = tf.layers.Dense(
            NUM_CLASSES,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.regularization_scale))
        logits = final_layer(final_embedding)  # (B, E) -> (B, C)

        tf.summary.histogram('kernel', final_layer.kernel)

        scores = tf.nn.sigmoid(logits, 'scores')
        prediction = tf.round(scores, 'predictions')  # we are only interested in scores anyway

    # Assign a default value to the train_op and loss to be passed for modes other than TRAIN
    loss = None
    train_op = None
    eval_metric_ops = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        main_loss = tf.losses.sigmoid_cross_entropy(labels.classes, logits)
        tf.summary.scalar('main_loss', main_loss)

        loss = tf.losses.get_total_loss()  # get sum of all explicitly defined losses and regularization losses

        learning_rate = params.learning_rate * 0.01 + tf.train.exponential_decay(
            params.learning_rate, global_step,
            12000, 0.5, staircase=False)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step)

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(
                labels.classes,
                prediction,
                name='accuracy')

            eval_metric_ops = {
                'accuracy': accuracy
            }

    prediction = {
        'id': features.id,
        'text': features.text,
        'text_length': features.text_length,
        'prediction': prediction,
        'scores': scores
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prediction,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
