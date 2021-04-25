import utils
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image

BATCH_SIZE = 100
HIDDEN_SIZE = 1024
EPOCHS_BETWEEN_EVAL = 1
MODEL_DIR = "./mnist_model_mtf"

MESH_SHAPE = "b1:2;b2:2"
LAYOUT = "row_blocks:b1;col_blocks:b2"


def mnist_model(image, labels, mesh):
    batch_dim = mtf.Dimension("batch", BATCH_SIZE)
    row_blocks_dim = mtf.Dimension("row_blocks", 4)
    col_blocks_dim = mtf.Dimension("col_blocks", 4)
    rows_dim = mtf.Dimension("rows_size", 7)
    cols_dim = mtf.Dimension("cols_size", 7)

    classes_dim = mtf.Dimension("classes", 10)
    one_channel_dim = mtf.Dimension("one_channel", 1)

    x = mtf.import_tf_tensor(
        mesh, tf.reshape(image, [BATCH_SIZE, 4, 7, 4, 7, 1]),
        mtf.Shape([batch_dim,
                   row_blocks_dim, rows_dim,
                   col_blocks_dim, cols_dim,
                   one_channel_dim]))
    x = mtf.transpose(x, [batch_dim,
                          row_blocks_dim, col_blocks_dim,
                          rows_dim, cols_dim,
                          one_channel_dim])

    # conv layers
    filters1_dim = mtf.Dimension("filters1", 16)
    filters2_dim = mtf.Dimension("filters2", 16)
    conv_layer0 = mtf.layers.conv2d_with_blocks(
        x, filters1_dim,
        filter_size=[9, 9], strides=[1, 1], padding="SAME",
        h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim,
        name="conv0")
    f1 = mtf.relu(conv_layer0)

    conv_layer1 = mtf.layers.conv2d_with_blocks(
        f1, filters2_dim,
        filter_size=[9, 9], strides=[1, 1], padding="SAME",
        h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim,
        name="conv1")
    f2 = mtf.relu(conv_layer1)

    x = mtf.reduce_mean(f2, reduced_dim=filters2_dim)

    # hidden layers
    hidden_dim1 = mtf.Dimension("hidden1", HIDDEN_SIZE)
    hidden_dim2 = mtf.Dimension("hidden2", HIDDEN_SIZE)

    h1 = mtf.layers.dense(
        x, hidden_dim1,
        reduced_dims=x.shape.dims[-4:],
        activation=mtf.relu,
        name="hidden1")

    h2 = mtf.layers.dense(
        h1, hidden_dim2,
        activation=mtf.relu,
        name="hidden2")

    logits = mtf.layers.dense(h2, classes_dim, name="logits")

    if labels is None:
        loss = None
    else:
        labels = mtf.import_tf_tensor(
            mesh, tf.reshape(labels, [BATCH_SIZE]), mtf.Shape([batch_dim]))
        loss = mtf.layers.softmax_cross_entropy_with_logits(
            logits, mtf.one_hot(labels, classes_dim), classes_dim)
        loss = mtf.reduce_mean(loss)
    return logits, loss


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    tf.logging.info("features = %s labels = %s mode = %s params=%s" % (features, labels, mode, params))
    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    logits, loss = mnist_model(features, labels, mesh)
    mesh_shape = mtf.convert_to_shape(MESH_SHAPE)
    layout_rules = mtf.convert_to_layout_rules(LAYOUT)
    mesh_size = mesh_shape.size
    mesh_devices = [""] * mesh_size
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        mesh_shape, layout_rules, mesh_devices)

    if mode == tf.estimator.ModeKeys.TRAIN:
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])
        optimizer = mtf.optimize.AdafactorOptimizer()
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    restore_hook = mtf.MtfRestoreHook(lowering)

    tf_logits = lowering.export_to_tf_tensor(logits)
    if mode != tf.estimator.ModeKeys.PREDICT:
        tf_loss = lowering.export_to_tf_tensor(loss)
        tf.summary.scalar("loss", tf_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))
        train_op = tf.group(tf_update_ops)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=10,
            keep_checkpoint_every_n_hours=2,
            defer_build=False, save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            MODEL_DIR,
            save_steps=1000,
            saver=saver,
            listeners=[saver_listener])

        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(tf_logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(tf_loss, "cross_entropy")
        tf.identity(accuracy[1], name="train_accuracy")

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar("train_accuracy", accuracy[1])

        # restore_hook must come before saver_hook
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
            training_chief_hooks=[restore_hook, saver_hook])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(tf_logits, axis=1),
            "probabilities": tf.nn.softmax(tf_logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            prediction_hooks=[restore_hook],
            export_outputs={
                "classify": tf.estimator.export.PredictOutput(predictions)
            })
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=tf_loss,
            evaluation_hooks=[restore_hook],
            eval_metric_ops={
                "accuracy":
                    tf.metrics.accuracy(
                        labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
            })


def preprocess_train_data():
    ds = utils.train_data()
    ds_batched = ds.cache().shuffle(buffer_size=50000).batch(BATCH_SIZE)
    ds = ds_batched.repeat(EPOCHS_BETWEEN_EVAL)
    return ds


def preprocess_test_data():
    ds = utils.test_data()
    ds_batched = ds.batch(BATCH_SIZE).make_one_shot_iterator()
    ds = ds_batched.get_next()
    return ds


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)

    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR)

    for _ in range(1):
        mnist_classifier.train(input_fn=preprocess_train_data, hooks=None)
        eval_result = mnist_classifier.evaluate(input_fn=preprocess_test_data)
        print(eval_result)
# mnist_model(, tf.convert_to_tensor(['elephant']), mesh)
