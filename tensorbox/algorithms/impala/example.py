import tensorflow as tf
from multiprocessing import Process, Queue
from time import sleep

import model
import numpy as np


def generate_noised_data(size=1000):
    t = np.arange(start=0.0, stop=2*np.pi, step=size)
    noise = np.random.random(size=size) * 0.1
    x = np.sin(t)
    y = x + noise
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    return x, y


def update_target_graph(source, target):
    """
    Copies values from one tf.graph() to another
    Useful for setting worker network parameters equal to global network.

    :param source: str()
    :param target: str()
    :return: tf.operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=source)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target)

    assert from_vars, 'update_target_graph: from_vars is an empty list, source={}'.format(source)
    assert to_vars, 'update_target_graph: to_vars is an empty list, target={}'.format(target)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Actor:
    def __init__(self, scope='actor'):
        self.scope = scope
        self.session = None
        with tf.variable_scope(scope):
            self.data_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='data_ph')
            self.labels_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels_ph')
            self.logits = model.build_mlp_graph(self.data_ph, layers=(128, 128))

    def train(self, x, y):
        raise NotImplementedError

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

    def predict(self, x):
        if self.session is None:
            self.session = tf.get_default_session()
        logits = self.session.run(self.logits,
                                         feed_dict={self.data_ph: x})
        return logits


class Learner(Actor):
    def __init__(self, scope='learner'):
        super().__init__(scope=scope)
        with tf.variable_scope(scope):
            self.global_step = tf.train.create_global_step()
            self.loss = tf.losses.mean_squared_error(labels=self.labels_ph, predictions=self.logits)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=5.0e-4)
            # params = tf.trainable_variables()
            # build gradients except for old policy nodes
            self.train_op = self.optimizer.minimize(self.loss)  # , global_step=self.global_step)

    def train(self, x, y):
        if self.session is None:
            self.session = tf.get_default_session()
        train_loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.data_ph: x, self.labels_ph: y})
        return train_loss


def learner(shared_job_device, queue):
    # Note:
    # the learner should be placed on a GPU for performance boost
    with tf.device(shared_job_device):
        learner = Learner()

    server = tf.train.Server(cluster,
                             job_name="learner",
                             task_index=0)
    max_steps = 10000

    with tf.Session(target=server.target) as sess:

        print("Parameter server: initializing variables...")
        print(tf.trainable_variables())
        sess.run(tf.global_variables_initializer())

        global_step = 0
        while global_step < max_steps:

            x, y = queue.get()
            # x, y = data
            loss = learner.train(x, y)
            if global_step % 1000 == 0:
                print('Loss =', loss)
            global_step += 1
            # print('new params:')
            # learner.get_values()
            # train_loss, _ = sess.run([loss, train_op],
            #                          feed_dict={data_ph: x, labels_ph: y})
            # print("Parameter server: var has value %.1f" % val)
            # sleep(1.0)

    print("Learner: request join...")
    server.join()


def actor(worker_n, shared_job_device, queue):
    local_job_device = '/job:worker/task:{}'.format(worker_n)

    with tf.device(local_job_device):
        actor = Actor()

    with tf.device(shared_job_device):
        learner = Learner()

    update_local_vars = update_target_graph(source='learner', target='actor')

    # start
    server = tf.train.Server(cluster,
                             job_name="worker",
                             task_index=worker_n)

    with tf.Session(target=server.target) as sess:
        # print('worker ({}) fetches values'.format(worker_n))
        print('worker: run global init, session:', sess)
        sess.run(tf.global_variables_initializer())

        # print("Worker %d: waiting for cluster connection..." % worker_n)
        # sess.run(tf.report_uninitialized_variables())
        # print("Worker %d: cluster ready!" % worker_n)
        # print(tf.report_uninitialized_variables())
        # while sess.run(tf.report_uninitialized_variables()):
        #     print("Worker %d: waiting for variable initialization..." % worker_n)
        #     sleep(1.0)
        # print("Worker %d: variables initialized" % worker_n)

        for i in range(5):
            print('worker ({}) fetches values'.format(worker_n))
            sess.run(update_local_vars)
            data = np.array([[np.pi]])
            print(actor.predict(x=data))

            # produce data
            for _ in range(1000):
                x, y = generate_noised_data()
                data = (x, y)
                queue.put(data)

            # sess.run(var_shared.assign_add(1.0))
            # print("Worker %d: copy shared var" % worker_n)

            sleep(0.2)

        # print('worker ({}): END var_local = {}'.format(worker_n, var_local.eval()))

    print("Worker ({}) requests join".format(worker_n))
    server.join()


if __name__ == '__main__':
    workers = ['localhost:{}'.format(3001 + i) for i in range(3)]
    jobname = 'impala'
    cluster = tf.train.ClusterSpec({
        "worker": workers,
        "learner": ["localhost:3000"]
    })

    print(cluster)
    shared_job_device = '/job:learner/task:0'

    queue = Queue(maxsize=100)

    processes = [Process(target=learner, args=(shared_job_device, queue), daemon=True)]  # add parameter server

    for w in range(len(workers)):  # create worker processes
        processes.append(Process(target=actor, args=(w, shared_job_device, queue), daemon=True))
        sleep(0.1)

    for p in processes:
        p.start()

    sleep(20)
    for p in processes:
        p.terminate()