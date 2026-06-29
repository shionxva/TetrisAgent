# Training Loop for the Tetris MLP


## What the model learns

The agent does not choose raw left/right/rotate actions step by step. Instead, for
each piece it evaluates every legal final placement and learns which placement is
best in the long run.

The four input features are:

1. cleared lines
2. holes
3. bumpiness
4. total height

Each possible placement is represented by these features, and the MLP predicts the
value of that placement.

## Training loop overview

The loop in `train.py` follows this pattern:

```text
initialize policy network and target network
initialize replay memory
for each episode:
    get every legal placement for the current piece
    score all placements with the policy network
    choose one placement with epsilon-greedy exploration
    apply the placement in the environment
    store the transition in replay memory

    if enough transitions have been collected and this step is a training step:
        sample a random batch from replay memory
        compute target values with the target network
        compute MSE loss between predicted Q-values and targets
        backpropagate, clip gradients, and update the policy network

    periodically copy policy network weights into the target network
    periodically save checkpoints and log metrics
```

## Step-by-step explanation

### 1. Setup

The script creates two copies of the MLP:

- `policy_net`: the network being trained
- `target_net`: a slower-moving copy used to compute stable targets

It also creates:

- an Adam optimizer
- an MSE loss function
- a replay buffer with a fixed maximum size
- a TensorBoard writer for logging

### 2. Action selection

At each turn, the environment returns all legal placements for the current piece.
The policy network scores each placement, and the trainer uses epsilon-greedy
selection:

- with probability `epsilon`, choose a random placement
- otherwise choose the highest-scoring placement

`epsilon` decays linearly over training, so the agent explores early and exploits
later.

### 3. Environment step

After selecting a placement, the trainer calls `env.step(action)`.
This returns:

- the reward for the placement
- whether the game ended

The chosen placement features, reward, and next possible placements are stored in
replay memory.

### 4. Replay-based learning

Training does not happen on every environment step. It starts only after the replay
buffer has enough samples, and then it runs every `train_every` steps.

For each mini-batch:

1. Sample random transitions from replay memory.
2. Predict Q-values for the chosen placements with `policy_net`.
3. Build target values with `target_net`.
4. Compute the Bellman target:

   `target = reward + gamma * max(next_q_values)`

   If the game is over, the target is just the reward.
5. Compute MSE loss between predicted values and targets.
6. Backpropagate the loss.
7. Clip gradients to keep training stable.
8. Apply the optimizer step.

### 5. Target network updates

The target network is copied from the policy network every
`target_update_interval` steps.
This keeps the target values stable while the policy network is changing.
