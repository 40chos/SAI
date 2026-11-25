import numpy as np
import tensorflow as tf

# Той самий алгоритм, але на сучасному API TF2 (tf.GradientTape замість Session)

np.random.seed(42)
tf.random.set_seed(42)

n_samples, batch_size, num_steps = 1000, 100, 20000

X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

k = tf.Variable(tf.random.normal((1, 1)), name='slope')
b = tf.Variable(tf.zeros((1,)), name='bias')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

display_step = 2000
for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch = X_data[indices]
    y_batch = y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X_batch, k) + b
        loss = tf.reduce_sum((y_batch - y_pred) ** 2)

    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (i + 1) % display_step == 0:
        print('Эпоха %d: %.8f, k=%.4f, b=%.4f' %
              (i + 1, loss.numpy(), k.numpy()[0][0], b.numpy()[0]))

print(f"\nІстинні параметри моделі: k=2.0, b=1.0")
print(f"Навчені параметри моделі (TF2/GradientTape): "
      f"k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")
