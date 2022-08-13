import tensorcircuit as tc
import optax
import jax.numpy as jnp
import jax
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

# dataset = 'mnist'
dataset = 'fashion'
readout_mode = 'softmax'
# readout_mode = 'sample'
encoding_mode = 'vanilla'
# encoding_mode = 'mean'
# encoding_mode = 'half'

n = 8
n_node = 8
k = 48

def filter(x, y, class_list):
    keep = jnp.zeros(len(y)).astype(bool)
    for c in class_list:
        keep = keep | (y == c)
    x, y = x[keep], y[keep]
    y = jax.nn.one_hot(y, n_node)
    return x, y

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
    y_test = jax.nn.one_hot(y_test, n_node)

    class_test_loss = []
    class_test_acc = []
    for n_class in jnp.arange(2, 9):
        world_test_loss = []
        world_test_acc = []
        for world in tqdm(range(n_world)):

            params_list = []
            opt_state_list = []
            data_list = []
            iter_list = []
            for node in range(n_node-1):
                x_train_node, y_train_node = filter(x_train, y_train, [(node+i)%n_node for i in range(n_class)])
                # x_train_node, y_train_node = x_train, jax.nn.one_hot(y_train, n_node)
                data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
                data_list.append(data)
                iter_list.append(iter(data))

                key, subkey = jax.random.split(key)
                params = jax.random.normal(subkey, (3 * k, n))
                opt = optax.adam(learning_rate=1e-2)
                opt_state = opt.init(params)
                params_list.append(params)
                opt_state_list.append(opt_state)

            loss_list = []
            acc_list = []
            for e in tqdm(range(5), leave=False):
                for b in range(100):
                    for node in range(n_node-1):
                        try:
                            x, y = next(iter_list[node])
                        except StopIteration:
                            iter_list[node] = iter(data_list[node])
                            x, y = next(iter_list[node])
                        x = x.numpy()
                        y = y.numpy()
                        loss_val, grad_val = compute_loss(params_list[node], x, y, k)
                        updates, opt_state_list[node] = opt.update(grad_val, opt_state_list[node], params_list[node])
                        params_list[node] = optax.apply_updates(params_list[node], updates)
                    
                    avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
                    for node in range(n_node-1):
                        params_list[node] = avg_params
                    
                    if b % 25 == 0:
                        avg_loss = jnp.mean(compute_loss(avg_params, x_test[:1024], y_test[:1024], k)[0])
                        loss_list.append(avg_loss)
                        acc_list.append(compute_accuracy(avg_params, x_test[:1024], y_test[:1024], k).mean())
                        tqdm.write(f"world {world}, epoch {e}, batch {b}/{100}: loss {avg_loss}, accuracy {acc_list[-1]}")

            test_acc = jnp.mean(pred(avg_params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
            test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:1024], k)) * y_test[:1024])

            world_test_loss.append(test_loss)
            world_test_acc.append(test_acc)
            tqdm.write(f"world {world}: test loss {test_loss}, test accuracy {test_acc}")
        avg_test_loss = jnp.mean(jnp.array(world_test_loss), axis=0)
        avg_test_acc = jnp.mean(jnp.array(world_test_acc), axis=0)
        std_test_loss = jnp.std(jnp.array(world_test_loss), axis=0)
        std_test_acc = jnp.std(jnp.array(world_test_acc), axis=0)
        print(f'n_class {n_class}, test loss: {avg_test_loss}+-{std_test_loss}, test acc: {avg_test_acc}+-{std_test_acc}')
        class_test_loss.append(world_test_loss)
        class_test_acc.append(world_test_acc)

    os.makedirs(f'./{dataset}/qFedAvg-noniid/', exist_ok=True) 
    jnp.save(f'./{dataset}/qFedAvg-noniid/test_loss.npy', class_test_loss)
    jnp.save(f'./{dataset}/qFedAvg-noniid/test_acc.npy', class_test_acc)

    
    
    