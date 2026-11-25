import numpy as np
import tensorflow as tf

# у TF2 старе API (placeholder, Session) доступне через tf.compat.v1
tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1

np.random.seed(42)
tf1.set_random_seed(42)

n_samples, batch_size, num_steps = 1000, 100, 20000

# Генерація вхідних даних: X рівномірно на [1, 10], y = 2x + 1 + шум N(0, 2)
X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

X = tf1.placeholder(tf.float32, shape=(batch_size, 1))
y = tf1.placeholder(tf.float32, shape=(batch_size, 1))

with tf1.variable_scope('linear-regression'):
    k = tf1.Variable(tf1.random_normal((1, 1)), name='slope')
    b = tf1.Variable(tf1.zeros((1,)), name='bias')

y_pred = tf1.matmul(X, k) + b
loss = tf1.reduce_sum((y - y_pred) ** 2)
# learning_rate=0.0001 підібрано дослідним шляхом - з більшим кроком (0.001)
# градієнт розходиться і loss, k, b стають NaN
optimizer = tf1.train.GradientDescentOptimizer(0.0001).minimize(loss)

display_step = 2000
with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())
    for i in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        _, loss_val, k_val, b_val = sess.run(
            [optimizer, loss, k, b], feed_dict={X: X_batch, y: y_batch})
        if (i + 1) % display_step == 0:
            print('Эпоха %d: %.8f, k=%.4f, b=%.4f' %
                  (i + 1, loss_val, k_val[0][0], b_val[0]))

    print(f"\nІстинні параметри моделі: k=2.0, b=1.0")
    print(f"Навчені параметри моделі: k={k_val[0][0]:.4f}, b={b_val[0]:.4f}")
