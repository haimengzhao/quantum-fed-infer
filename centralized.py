import tensorcircuit as tc
import optax
import jax.numpy as jnp
import jax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.mixture import GaussianMixture

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rc('font', size=14)

K = tc.set_backend('jax')
key = jax.random.PRNGKey(42)
tf.random.set_seed(42)

n_world = 10

dataset = 'mnist'
# dataset = 'fashion'
# readout_mode = 'softmax'
readout_mode = 'sample'
encoding_mode = 'vanilla'
# encoding_mode = 'mean'
# encoding_mode = 'half'

n = 8
n_node = 8
k = 48

def clf(params, c, k):
    for j in range(k):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c

def readout(c):
    if readout_mode == 'softmax':
        logits = []
        for i in range(n_node):
            logits.append(jnp.real(c.expectation([tc.gates.z(), [i,]])))
        logits = jnp.stack(logits, axis=-1) * 10
        probs = jax.nn.softmax(logits)
    elif readout_mode == 'sample':
        wf = jnp.abs(c.wavefunction()[:n_node])**2
        probs = wf / jnp.sum(wf)
    return probs

def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))
loss = K.jit(loss, static_argnums=[3])

def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)
accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])

def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs
pred = K.vmap(pred, vectorized_argnums=[1])

if __name__ == '__main__':
    # numpy data
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    ind = y_test == 9
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_test == 8
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_train == 9
    x_train, y_train = x_train[~ind], y_train[~ind]
    ind = y_train == 8
    x_train, y_train = x_train[~ind], y_train[~ind]

    x_train = x_train / 255.0
    if encoding_mode == 'vanilla':
        mean = 0
    elif encoding_mode == 'mean':
        mean = jnp.mean(x_train, axis=0)
    elif encoding_mode == 'half':
        mean = 0.5
    x_train = x_train - mean
    x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2**(n/2)), int(2**(n/2)))).numpy()[..., 0].reshape(-1, 2**n)
    x_train = x_train / jnp.sqrt(jnp.sum(x_train**2, axis=-1, keepdims=True))

    x_test = x_test / 255.0
    x_test = x_test - mean
    x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2**(n/2)), int(2**(n/2)))).numpy()[..., 0].reshape(-1, 2**n)
    x_test = x_test / jnp.sqrt(jnp.sum(x_test**2, axis=-1, keepdims=True))

    y_train = jax.nn.one_hot(y_train, n_node)
    y_test = jax.nn.one_hot(y_test, n_node)

    data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)

    world_train_loss = []
    world_test_loss = []
    world_train_acc = []
    world_test_acc = []
    for world in tqdm(range(n_world)):
        key, subkey = jax.random.split(key)
        params = jax.random.normal(subkey, (3 * k, n))
        opt = optax.adam(learning_rate=1e-2)
        opt_state = opt.init(params)

        loss_list = []
        loss_list_train = []
        acc_list = []
        acc_list_train = []
        for e in tqdm(range(3), leave=False):
            for i, (x, y) in enumerate(data):
                x = x.numpy()
                y = y.numpy()
                loss_val, grad_val = compute_loss(params, x, y, k)
                updates, opt_state = opt.update(grad_val, opt_state, params)
                params = optax.apply_updates(params, updates)
                loss_mean = jnp.mean(loss_val)
                if i % 50 == 0:
                    acc = jnp.mean(compute_accuracy(params, x, y, k))
                    acc_list_train.append(acc)
                    loss_list_train.append(loss_mean)
                    loss_mean = jnp.mean(compute_loss(params, x_test[:1024], y_test[:1024], k)[0])
                    acc = jnp.mean(compute_accuracy(params, x_test[:1024], y_test[:1024], k))
                    acc_list.append(acc)
                    loss_list.append(loss_mean)
                    print(f'world {world}, epoch {e}, {i}/{len(data)}: loss={loss_mean:.4f}, acc={acc:.4f}')

        test_acc = jnp.mean(pred(params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
        test_loss = -jnp.mean(jnp.log(pred(params, x_test[:1024], k)) * y_test[:1024])

        world_train_loss.append(loss_list)
        world_test_loss.append(test_loss)
        world_train_acc.append(acc_list)
        world_test_acc.append(test_acc)
        tqdm.write(f"world {world}: test loss {test_loss}, test accuracy {test_acc}")

    os.makedirs(f'./{dataset}/central/', exist_ok=True) 
    jnp.save(f'./{dataset}/central/train_loss.npy', world_train_loss)
    jnp.save(f'./{dataset}/central/train_acc.npy', world_train_acc)
    jnp.save(f'./{dataset}/central/test_loss.npy', world_test_loss)
    jnp.save(f'./{dataset}/central/test_acc.npy', world_test_acc)

    avg_test_loss = jnp.mean(jnp.array(world_test_loss), axis=0)
    avg_test_acc = jnp.mean(jnp.array(world_test_acc), axis=0)
    std_test_loss = jnp.std(jnp.array(world_test_loss), axis=0)
    std_test_acc = jnp.std(jnp.array(world_test_acc), axis=0)
    print(f'test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')