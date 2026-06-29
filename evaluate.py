"""
evaluate.py — greedy (epsilon=0) evaluation of a trained Tetris DQN.

Plays N full games with the learned value function, choosing the highest-valued
afterstate every move, and reports the distribution of cleared lines / pieces /
score. Writes a CSV of per-game results and a box-plot — exactly the held-out
evaluation the report should cite instead of noisy training-time numbers.

Usage:
    python evaluate.py --model trained_models/tetris --games 100
    python evaluate.py --model trained_models/tetris.pth --games 100 --env tetrisV2
"""
import argparse, csv, os
import numpy as np
import torch

def load_model(path):
    """Accept either a full pickled model or a state-dict checkpoint."""
    from src.deep_q_network import DeepQNetwork
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:  # checkpoint dict produced by new.py
        model = DeepQNetwork()
        sd = obj.get("policy_net", obj)
        model.load_state_dict(sd)
    model.eval()
    return model

def play_one(env, model):
    state = env.reset()
    while True:
        next_steps = env.get_next_states()
        actions, states = zip(*next_steps.items())
        states = torch.stack(states)
        with torch.no_grad():
            preds = model(states)[:, 0]
        action = actions[int(torch.argmax(preds).item())]
        _, done = env.step(action, render=False)
        if done:
            break
    return env.cleared_lines, env.tetrominoes, env.score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trained_models/tetris")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--env", default="tetris", choices=["tetris", "tetrisV2", "tetrisV3", "tetrisV4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    import random; random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    Tetris = __import__(f"src.{args.env}", fromlist=["Tetris"]).Tetris
    model = load_model(args.model)

    rows = []
    for g in range(args.games):
        lines, pieces, score = play_one(Tetris(), model)
        rows.append((g, lines, pieces, score))
        print(f"game {g+1:3d}/{args.games}: lines={lines:6d} pieces={pieces:7d} score={score}")

    lines = np.array([r[1] for r in rows], float)
    pieces = np.array([r[2] for r in rows], float)
    def stat(a): return dict(mean=a.mean(), median=np.median(a), std=a.std(), min=a.min(), max=a.max())
    print("\n=== Summary over", args.games, "games (greedy, eps=0) ===")
    print("lines :", {k: round(v,1) for k,v in stat(lines).items()})
    print("pieces:", {k: round(v,1) for k,v in stat(pieces).items()})

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["game","lines","pieces","score"]); w.writerows(rows)
    print("wrote", args.out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8,3.4))
        ax[0].boxplot(lines, vert=True, labels=["lines"]); ax[0].set_title("Cleared lines"); ax[0].grid(alpha=.3)
        ax[1].boxplot(pieces, vert=True, labels=["pieces"]); ax[1].set_title("Pieces placed"); ax[1].grid(alpha=.3)
        fig.suptitle(f"Greedy evaluation — {args.env}, {args.games} games")
        fig.tight_layout(); fig.savefig("eval_distribution.png", dpi=130)
        print("wrote eval_distribution.png")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
