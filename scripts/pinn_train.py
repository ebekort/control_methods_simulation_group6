'''
Train the LNN on the data written by pd+gravity_control_data.py

For every transition (x_k, u_k) -> x_k+1 the learned dynamics are integrated
one sample period with RK4 and compared with the measured next state (Eq. 8).
Adam minimises that error on the train trajectories, and the test
trajectories are evaluated after every epoch.

The error of every state component is divided by the standard deviation of
its one-step change, so joints with small signals (the wrist) count as much
as the big ones. The loss of predicting no change is printed as a reference.

    python scripts/pinn_train.py data_100_wide --epochs 20
    python scripts/pinn_train.py data_100 --out model_100hz.pkl
    python scripts/pinn_train.py data_100_wide --actuation network
    python scripts/pinn_train.py data_dither --actuation network --a-reg 0.01

The result is a pickle file with the parameters, loss weights, dt and the
training history. Only load pickle files you made yourself.
'''
import argparse
import pickle
import time
import jax
import numpy as np
import optax
from cli_utils import HelpFormatter
from pinn_data import batches, load_pairs
from pinn_model import N, actuation_penalty, init_params, loss_fn


def make_train_step(optimizer, dt, weights, a_reg):
    '''jit-compiled step: loss, gradients, Adam update'''

    def total_loss(params, x, u, x_next):
        loss = loss_fn(params, x, u, x_next, dt, weights)
        if a_reg:
            loss += a_reg * actuation_penalty(params, x[:, :N])
        return loss

    @jax.jit
    def train_step(params, opt_state, x, u, x_next):
        loss, grads = jax.value_and_grad(total_loss)(
            params, x, u, x_next
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return train_step


def make_evaluate(dt, weights, batch_size):
    '''Mean loss over a whole data set, computed in chunks'''
    loss = jax.jit(
        lambda params, x, u, x_next: loss_fn(
            params, x, u, x_next, dt, weights
        )
    )

    def evaluate(params, x, u, x_next):
        total = 0.0
        for i in range(0, len(x), batch_size):
            part = slice(i, i + batch_size)
            n = len(x[part])
            total += float(loss(params, x[part], u[part], x_next[part])) * n
        return total / len(x)

    return evaluate


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    ap.add_argument('folder', help='data folder, e.g. data_100_wide')
    ap.add_argument('--out', default='model.pkl', help='output file')
    ap.add_argument(
        '--epochs', type=int, default=20, help='passes over the data'
    )
    ap.add_argument(
        '--batch-size', type=int, default=256, help='samples per step'
    )
    ap.add_argument(
        '--lr', type=float, default=1e-3, help='peak learning rate'
    )
    ap.add_argument(
        '--hidden', type=int, nargs='+', default=[64, 64],
        help='hidden layer widths of each network'
    )
    ap.add_argument(
        '--actuation', choices=('network', 'diagonal', 'identity'),
        default='diagonal',
        help='how the actuation matrix A is learned (see pinn_model.py)'
    )
    ap.add_argument(
        '--a-reg', type=float, default=0.0,
        help='weight of the penalty pulling A towards the identity'
    )
    ap.add_argument(
        '--mass-init', choices=('flat', 'ramp'), default='flat',
        help='starting mass matrix (see init_params in pinn_model.py)'
    )
    ap.add_argument('--seed', type=int, default=0, help='random seed')
    args = ap.parse_args()

    x, u, x_next, dt = load_pairs(args.folder, 'train')
    x_te, u_te, x_next_te, dt_te = load_pairs(args.folder, 'test')
    if abs(dt - dt_te) > 1e-9:
        raise ValueError('train and test data have a different dt')
    print(f'{len(x)} train pairs, {len(x_te)} test pairs, dt = {dt:g} s')

    # weight each state component by 1 / (std of its one-step change)^2
    weights = 1.0 / np.maximum((x_next - x).std(axis=0), 1e-8) ** 2
    baseline = np.mean(np.sum(weights * (x_next_te - x_te) ** 2, axis=1))
    print(f'test loss when predicting no change: {baseline:.4f}')

    rng = np.random.default_rng(args.seed)
    params = init_params(
        jax.random.PRNGKey(args.seed), tuple(args.hidden), args.actuation,
        args.mass_init
    )
    steps_per_epoch = len(x) // args.batch_size
    optimizer = optax.adam(
        optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch)
    )
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer, dt, weights, args.a_reg)
    evaluate = make_evaluate(dt, weights, args.batch_size)

    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        losses = []
        for xb, ub, xnb in batches(x, u, x_next, args.batch_size, rng):
            params, opt_state, loss = train_step(
                params, opt_state, xb, ub, xnb
            )
            losses.append(loss)  # float() here would sync the GPU every step
        train_loss = float(np.mean(jax.device_get(losses)))
        test_loss = evaluate(params, x_te, u_te, x_next_te)
        history.append((epoch, train_loss, test_loss))
        print(
            f'epoch {epoch:3d}  train {train_loss:.4f}  '
            f'test {test_loss:.4f}  ({time.time() - start:.1f} s)'
        )

    result = {
        'params': jax.device_get(params), 'weights': weights, 'dt': dt,
        'hidden': tuple(args.hidden), 'actuation': args.actuation,
        'a_reg': args.a_reg, 'mass_init': args.mass_init,
        'history': history,
    }
    with open(args.out, 'wb') as fh:
        pickle.dump(result, fh)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
