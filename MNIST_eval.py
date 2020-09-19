import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]

    for keys in keys_list:
        FLAGS.__delattr__(keys)

# mnist 데이터 가져오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

# Train된 모델들 찾아서 가져오기
filenames = os.listdir(os.path.join(os.path.curdir, 'runs'))

file = open('./eval_log.txt', 'w')
# 모델이 저장된 checkpoint 경로
for filename in filenames:
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()
    tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{filename}/checkpoints", "Checkpoint directory from training run")

    FLAGS = tf.flags.FLAGS
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) #가장 validation accuracy가 높은 시점 load
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph(f"{checkpoint_file}.meta")
            saver.restore(sess, checkpoint_file) # 저장했던 모델 load

            # Get the placeholders from the graph by name, name을 통해 operation 가져오기
            X = graph.get_operation_by_name("X").outputs[0]
            Y = graph.get_operation_by_name("Y").outputs[0]
            # TODO *hint* dropout 확률을 placeholder에 추가
            dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

            hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]
            correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            feed_dict = {X:mnist.test.images, Y:mnist.test.labels, dropout_prob: 0.0}

            test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            file.write(f'{filename}\t{test_accuracy}\n')
            print('- Timestamp        : ', filename)
            print('- Test Max Accuracy:', test_accuracy)
    sess.close()
file.close()



