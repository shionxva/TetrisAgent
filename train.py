"""
Improved placement-based DQN training for Tetris.

Run in Colab from the project folder:
    python train.py --num_epochs 3000

Resume:
    python train.py --resume trained_models/tetris_checkpoint_1000.pth --num_epochs 3000
"""

import argparse
import os
import random
import shutil
from collections import deque

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter
    except Exception:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def close(self):
                pass


from src.deep_q_network import DeepQNetwork
from src.tetris_V2 import Tetris


def get_args():
    parser = argparse.ArgumentParser("Improved Deep Q Network training for Tetris")

    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)

    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=500)

    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--train_start", type=int, default=3000)
    parser.add_argument("--target_update_interval", type=int, default=1000)
    parser.add_argument("--train_every", type=int, default=4, help="Run one optimizer update every N environment steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Write TensorBoard loss every N environment steps")

    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    # In Colab, keep render off. Rendering uses cv2.imshow and will usually fail/slow down.
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def get_epsilon(epoch, opt):
    return opt.final_epsilon + max(opt.num_decay_epochs - epoch, 0) * (
        opt.initial_epsilon - opt.final_epsilon
    ) / opt.num_decay_epochs


def compute_targets(batch, target_net, device, gamma):
    """Vectorized Bellman target calculation.

    Old version called target_net once per replay item. That creates hundreds of
    tiny CUDA calls per optimizer step. This version concatenates all next states,
    does one forward pass, then splits the result back per replay item.
    """
    rewards = torch.tensor(
        [transition[1] for transition in batch],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    targets = rewards.clone()

    next_state_chunks = []
    next_state_lengths = []
    batch_indices = []

    for batch_idx, (_, _, next_possible_states, done) in enumerate(batch):
        if not done and next_possible_states.numel() > 0:
            next_state_chunks.append(next_possible_states)
            next_state_lengths.append(next_possible_states.shape[0])
            batch_indices.append(batch_idx)

    if next_state_chunks:
        with torch.no_grad():
            all_next_states = torch.cat(next_state_chunks, dim=0).to(device, non_blocking=True)
            all_next_q = target_net(all_next_states).squeeze(1)
            split_next_q = torch.split(all_next_q, next_state_lengths)

            for batch_idx, q_values in zip(batch_indices, split_next_q):
                targets[batch_idx, 0] += gamma * q_values.max()

    return targets


def save_checkpoint(path, policy_net, target_net, optimizer, epoch, global_step, epsilon):
    checkpoint = {
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "epsilon": epsilon,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, policy_net, target_net, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    return epoch, global_step


def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    writer = SummaryWriter(opt.log_path)

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    policy_net = DeepQNetwork().to(device)
    target_net = DeepQNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    epoch = 0
    global_step = 0

    if opt.resume is not None:
        epoch, global_step = load_checkpoint(opt.resume, policy_net, target_net, optimizer, device)
        print(f"Resumed from {opt.resume}")
        print(f"Start epoch: {epoch}, global step: {global_step}")

    replay_memory = deque(maxlen=opt.replay_memory_size)
    env.reset()

    while epoch < opt.num_epochs:
        epsilon = get_epsilon(epoch, opt)

        # All possible final placements for the current piece.
        next_steps = env.get_next_states()
        if len(next_steps) == 0:
            env.reset()
            continue

        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        policy_net.eval()
        with torch.no_grad():
            q_predictions = policy_net(next_states)[:, 0]
        policy_net.train()

        if random.random() <= epsilon:
            selected_index = random.randint(0, len(next_actions) - 1)
        else:
            selected_index = torch.argmax(q_predictions).item()

        action = next_actions[selected_index]
        selected_state_action_feature = next_states[selected_index].detach().cpu()

        reward, done = env.step(action, render=opt.render)
        global_step += 1

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines

            replay_memory.append([
                selected_state_action_feature,
                reward,
                torch.empty(0),
                True,
            ])

            env.reset()
            epoch += 1

            print(
                f"Epoch: {epoch}/{opt.num_epochs}, "
                f"Score: {final_score}, "
                f"Tetrominoes: {final_tetrominoes}, "
                f"Cleared lines: {final_cleared_lines}, "
                f"Epsilon: {epsilon:.5f}, "
                f"Replay: {len(replay_memory)}"
            )

            writer.add_scalar("Train/Score", final_score, epoch)
            writer.add_scalar("Train/Tetrominoes", final_tetrominoes, epoch)
            writer.add_scalar("Train/Cleared lines", final_cleared_lines, epoch)
            writer.add_scalar("Train/Epsilon", epsilon, epoch)

            if epoch > 0 and epoch % opt.save_interval == 0:
                checkpoint_path = os.path.join(opt.saved_path, f"tetris_checkpoint_{epoch}.pth")
                save_checkpoint(
                    checkpoint_path,
                    policy_net,
                    target_net,
                    optimizer,
                    epoch,
                    global_step,
                    epsilon,
                )
                torch.save(policy_net, os.path.join(opt.saved_path, f"tetris_{epoch}"))

        else:
            future_steps = env.get_next_states()
            if len(future_steps) > 0:
                _, future_states = zip(*future_steps.items())
                next_possible_states = torch.stack(future_states).detach().cpu()
                terminal = False
            else:
                next_possible_states = torch.empty(0)
                terminal = True

            replay_memory.append([
                selected_state_action_feature,
                reward,
                next_possible_states,
                terminal,
            ])

        if len(replay_memory) < opt.train_start or global_step % opt.train_every != 0:
            continue

        batch = random.sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_action_batch, _, _, _ = zip(*batch)
        state_action_batch = torch.stack(state_action_batch).to(device, non_blocking=True)

        q_values = policy_net(state_action_batch)
        y_batch = compute_targets(batch, target_net, device, opt.gamma)

        loss = criterion(q_values, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
        optimizer.step()

        if global_step % opt.log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)

        if global_step % opt.target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

    final_checkpoint_path = os.path.join(opt.saved_path, "tetris_final_checkpoint.pth")
    save_checkpoint(
        final_checkpoint_path,
        policy_net,
        target_net,
        optimizer,
        epoch,
        global_step,
        get_epsilon(epoch, opt),
    )
    torch.save(policy_net, os.path.join(opt.saved_path, "tetris"))
    writer.close()

if __name__ == "__main__":
    train(get_args())
