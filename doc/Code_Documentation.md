# Code Documentation — `tetrisV2.py` and `train.py`

This document explains the two core files of the project:

- **`src/tetrisV2.py`** — the Tetris **environment** (the game + the shaped reward).
- **`train.py`** — the **Deep Q-Learning trainer** that learns to play it.

---

## 1. How the two files work together (the pipeline)

The agent never controls the piece move-by-move. Instead, each turn follows this loop:

```
            ┌─────────────────────── train.py ───────────────────────┐
            │                                                            │
  env.get_next_states()  →  network scores every placement  →  ε-greedy pick
            │                                                            │
            └──→ env.step(action) ──→ reward + done ──→ store in replay buffer
                                                            │
                          sample batch ──→ compute target (r + γ·maxV) ──→ MSE loss
                                                            │
                                          update policy_net; sync target_net
```

1. **`tetrisV2.py`** exposes the board as a set of **afterstates**: for the current piece it lists every legal final placement and, for each, a 4-number summary of the resulting board.
2. **`train.py`** feeds those summaries to a neural network, picks the best placement (mostly), and calls `env.step()` to play it.
3. `step()` returns a **shaped reward** and whether the game ended.
4. The trainer stores the experience, and every few steps trains the network so its value estimates match the rewards it actually receives.

The state passed between them is always a **4-feature vector**: `[lines_cleared, holes, bumpiness, total_height]`.

---

## 2. `src/tetrisV2.py` — the environment

### Board representation
- `self.board` is a 2-D list of `height` rows × `width` columns (default 20 × 10).
- Each cell is `0` (empty) or `1–7` (a colour id identifying which piece filled it).
- `self.pieces` holds the 7 tetrominoes as small 2-D grids: index `0=O, 1=T, 2=S, 3=Z, 4=I, 5=J, 6=L`.

### Key instance variables
| Variable | Meaning |
|----------|---------|
| `self.board` | current locked board (falling piece not included) |
| `self.piece` | the current falling piece grid |
| `self.ind` | id (0–6) of the current piece |
| `self.bag` | remaining pieces in the current 7-bag (shuffled) |
| `self.current_pos` | `{"x", "y"}` top-left position of the falling piece |
| `self.score` | accumulated reward this game |
| `self.tetrominoes` | pieces placed this game (survival length) |
| `self.cleared_lines` | total lines cleared this game |
| `self.gameover` | `True` once the stack tops out |

### Function reference

**`__init__(self, height=20, width=10, block_size=20)`**
Sets the board dimensions, builds the side-panel image used for rendering (`extra_board`), then calls `reset()`. *Returns:* nothing (constructor). *Note:* the trainer calls it with keyword arguments, so argument order doesn't matter.

**`reset(self)`**
Starts a new game: empties the board, zeroes the counters, shuffles a fresh 7-bag, draws the first piece and centres it, sets `gameover=False`.
*Returns:* the initial 4-feature state vector (`get_state_properties` of the empty board).
*Logic:* the 7-bag (`list(range(7))` shuffled) guarantees each piece appears once before any repeats.

**`rotate(self, piece)`**
Rotates a piece grid 90° clockwise.
*Returns:* a new rotated 2-D list (does not modify the input).
*Logic:* standard matrix transpose-and-reverse implemented with index arithmetic.

**`get_state_properties(self, board)`**
Turns a board into the agent's state vector.
*Returns:* `torch.FloatTensor([lines_cleared, holes, bumpiness, height])`.
*Logic:* first clears any full rows (so the features describe the board *after* clears), then measures holes, bumpiness, and total height. **This is the only place the 4 features are defined**, so it is the heart of the data representation.

**`get_holes(self, board)`**
Counts covered empty cells.
*Returns:* integer number of holes.
*Logic:* for each column, skip the empty cells at the top until the first filled cell, then count every empty cell below it — those are holes trapped under blocks.

**`get_bumpiness_and_height(self, board)`**
Measures surface roughness and total stack height.
*Returns:* `(total_bumpiness, total_height)`.
*Logic:* per-column height = `board_height − index of the topmost filled cell`. `total_height` is their sum; `bumpiness` is the sum of absolute differences between neighbouring column heights (a flat surface = low bumpiness).

**`get_next_states(self)`**  ← *most important method*
Enumerates **every legal final placement** of the current piece.
*Returns:* a dict mapping `(x, rotation_index) → 4-feature tensor` of the resulting board.
*Logic:* the number of distinct rotations is piece-dependent (O = 1; S, Z, I = 2; T, J, L = 4) to avoid duplicates. For each rotation and each horizontal offset `x`, it copies the piece, "hard-drops" it (`while not check_collision: y += 1`), trims overflow (`truncate`), writes it onto a board copy (`store`), and records that board's features. The trainer uses this dict as its action set.

**`get_current_board_state(self)`**
Overlays the falling piece onto a copy of the board (for display only).
*Returns:* a 2-D board list with the active piece drawn in.

**`new_piece(self)`**
Spawns the next piece. Refills and reshuffles the bag if empty, pops the next id, centres it. If it immediately collides, sets `gameover=True` (top-out).
*Returns:* nothing (mutates state).

**`check_collision(self, piece, pos)`**
Tests whether the piece would collide **if it moved one row down** from `pos`.
*Returns:* `True` if it would hit the floor or an existing block, else `False`.
*Logic:* checks `future_y = pos["y"] + 1` against the board bounds and occupied cells. Used to find where a hard-drop stops.

**`truncate(self, piece, pos)`**
Handles the edge case where a piece locks partly above the top of the board.
*Returns:* `True` if this caused a game over, else `False`.
*Logic:* finds the lowest overlapping row; if the piece can't fit, it deletes its top rows one at a time, signalling overflow.

**`store(self, piece, pos)`**
Writes a piece's cells onto a **copy** of the board at `pos`.
*Returns:* the new board (original board untouched, so it's safe to call during enumeration).

**`check_cleared_rows(self, board)`**
Finds and removes completed rows.
*Returns:* `(number_of_cleared_lines, new_board)`.
*Logic:* scans from the bottom up; any row with no `0` is full and is removed via `remove_row`.

**`remove_row(self, board, indices)`**
Deletes the given full rows and adds empty rows at the top to keep the height constant.
*Returns:* the rebuilt board.

**`step(self, action, render=True, video=None)`**  ← *applies a move + computes reward*
Executes one chosen placement and advances the game.
*Returns:* `(reward, gameover)`.
*Logic, in order:*
1. Record `old_holes`, `old_bumpiness`, `old_height` (board **before** the move).
2. Unpack `action = (x, num_rotations)`, rotate the piece that many times, set its column.
3. Hard-drop: increase `y` until `check_collision`.
4. `truncate` → if overflow, `gameover = True`.
5. `store` the piece, then `check_cleared_rows` to clear full lines.
6. Recompute `new_holes/bumpiness/height` → compute the **deltas** (change caused by this move) and `max_height`.
7. **Shaped reward:**
   ```
   score = 40·lines_cleared − 8·Δholes − 2·Δbumpiness − 1·Δheight
   if max_height ≥ 16:  score −= 5      # discourage dangerously tall stacks
   score += 0.1                         # small survival bonus
   ```
8. Update `self.score`, `self.tetrominoes`, `self.cleared_lines`; spawn the next piece.
9. If the game ended, **override** the reward to `−100` (large terminal penalty).

> **Why deltas, not absolute values?** Penalising the *change* keeps survival worthwhile. A persistent per-step penalty on the board state would, under discounting, make "die immediately" the optimal strategy.

**`render(self, video=None)`**
Draws the board and a side panel (Score / Pieces / Lines) with OpenCV and PIL; optionally writes a frame to a video.
*Returns:* nothing (only used for visualisation/demo; training runs with `render=False`).

---

## 3. `train.py` — the DQN trainer

This file turns the environment into a learning loop. The network being trained is `DeepQNetwork` (`src/deep_q_network.py`): a small MLP `4 → 64 → 64 → 1` that maps a board's 4 features to a single value.

### Function reference

**`get_args()`**
Defines and parses all command-line options.
*Returns:* an `argparse` namespace.
*Key arguments:*
| Arg | Default | Meaning |
|-----|---------|---------|
| `--env` | `tetrisV2` | which environment/reward to use |
| `--num_epochs` | 3000 | number of **games** to play |
| `--num_decay_epochs` | 2000 | epochs over which ε decays |
| `--batch_size` | 256 | replay sample size per update |
| `--lr` | 1e-3 | Adam learning rate |
| `--gamma` | 0.99 | discount factor |
| `--replay_memory_size` | 30000 | replay buffer capacity |
| `--train_start` | 3000 | min transitions before training begins |
| `--train_every` | 4 | one optimizer update every N env steps |
| `--target_update_interval` | 1000 | sync target net every N steps |
| `--save_interval` | 500 | checkpoint every N epochs |
| `--log_path`, `--saved_path` | runs/v3, trained_models/v3 | output folders |
| `--resume` | None | checkpoint to continue from |

**`get_epsilon(epoch, opt)`**
Computes the current exploration rate.
*Returns:* a float ε.
*Logic:* linear decay from `initial_epsilon` (1.0) to `final_epsilon` (1e-3) across `num_decay_epochs`, then stays at the minimum. High ε early = explore; low ε later = exploit.

**`compute_targets(batch, target_net, device, gamma)`**
Builds the Q-learning regression targets for a batch.
*Returns:* a `[batch, 1]` tensor of target values.
*Logic:* for each stored transition `[feature, reward, next_possible_states, done]`:
- if `done` → target = `reward` (no future).
- else → target = `reward + γ · max(target_net(next_possible_states))` — i.e. reward plus the value of the **best** placement available for the next piece.
For speed it concatenates every transition's `next_possible_states` into one tensor, does a **single** forward pass through the target net, then splits the results back per transition. This is the true greedy Bellman backup `V(s) ← r + γ·max_a' V(s')`.

**`save_checkpoint(path, policy_net, target_net, optimizer, epoch, global_step, epsilon)`**
Saves a resumable checkpoint dict (both networks, optimizer state, and counters) to disk.
*Returns:* nothing.

**`_adapt_state_dict(sd, model)`**
Makes a saved `state_dict` compatible with the current network definition.
*Returns:* a remapped `state_dict`.
*Logic:* handles the `out.weight ↔ out.0.weight` naming difference (plain `Linear` vs `Sequential(Linear)`), so checkpoints saved by a slightly different model version still load.

**`load_checkpoint(path, policy_net, target_net, optimizer, device)`**
Restores a checkpoint for resuming.
*Returns:* `(epoch, global_step)` to continue from.
*Logic:* loads both networks through `_adapt_state_dict`; restores the optimizer inside a `try/except` (a mismatched optimizer state just warns and continues with a fresh one); reads back the saved counters.

**`train(opt)`**  ← *the main training loop*
Runs the whole DQN training procedure. *Returns:* nothing (writes logs + model files).
*Step by step:*
1. **Setup:** pick device (CUDA if available), set seeds, clear/create the log and model folders, open TensorBoard, **dynamically import** the chosen env (`src.<--env>`).
2. **Networks:** create `policy_net` (trained online) and `target_net` (a frozen copy used for stable targets); Adam optimizer; MSE loss. Optionally `resume`.
3. **Play loop** (`while epoch < num_epochs`):
   - `epsilon = get_epsilon(epoch)`.
   - `next_steps = env.get_next_states()` → all placements of the current piece.
   - Score them with `policy_net`; with prob. ε pick a **random** placement, else the **highest-valued** one.
   - Remember the chosen placement's features (`selected_state_action_feature`).
   - `reward, done = env.step(action)`; `global_step += 1`.
   - **If the game ended:** record final score/pieces/lines, store a terminal transition `[feature, reward, empty, True]`, reset the env, `epoch += 1`, print progress, log to TensorBoard, and checkpoint every `save_interval` epochs.
   - **Otherwise:** look up the **next** piece's placements (`future_steps`) and store the transition `[feature, reward, next_possible_states, terminal]`.
   - **Train** only if the buffer has ≥ `train_start` transitions **and** this is a `train_every`-th step:
     - sample a batch, compute `q_values = policy_net(chosen features)`,
     - `y_batch = compute_targets(...)`,
     - `loss = MSE(q_values, y_batch)`, backprop, **clip gradients** to norm 10, `optimizer.step()`.
   - Log the loss every `log_interval` steps; **sync** `target_net ← policy_net` every `target_update_interval` steps.
4. **Finish:** save a final checkpoint and the full model (`trained_models/v3/tetris`), close the writer.

### Important details
- **`epoch` = one full game**, so `num_epochs` is the number of games; **`global_step` = one placement**, so there are many steps per epoch.
- The **stored state is the afterstate** of the move taken, and **`next_possible_states`** is the afterstate set of the *next* piece — that's what makes the `max` in `compute_targets` a proper greedy lookahead inside the value function.
- The **target network** is updated only every 1,000 steps, which stabilises training (the network isn't chasing a constantly-moving target).
- Training starts only after 3,000 transitions are collected, so early batches aren't dominated by a few correlated games.

---

## 4. Summary

| File | Role | Core method | Produces |
|------|------|-------------|----------|
| `tetrisV2.py` | Environment + shaped reward | `get_next_states()`, `step()` | afterstate features, reward |
| `train.py` | DQN training loop | `train()`, `compute_targets()` | trained model + TensorBoard logs |

Together they implement a **placement-based Deep Q-Network**: the environment proposes every move and scores the resulting board with 4 features; the trainer learns a value function over those boards so the agent can pick the placement that leads to the best long-term outcome.

---

## 5. Demo And Agent Evaluation

| File | Function |
|------|----------------|
| `tetrisgame.py` | Full agent-controlled gameplay demo |
| `tetrisgame_human.py` | Press `T` to toggle AI on/off |
| `evaluate.py` | Agent evaluation script |
