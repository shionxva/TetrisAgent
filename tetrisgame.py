from src.tetris_V2 import Tetris
import torch
import time
from src.deep_q_network import DeepQNetwork


class AIController:
    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

    def choose_action(self, env):

        states = env.get_next_states()

        if len(states) == 0:
            return None

        actions, features = zip(*states.items())

        features = torch.stack(features).to(self.device)

        with torch.no_grad():
            q_values = self.model(features).squeeze(1)

        best_idx = torch.argmax(q_values).item()

        return actions[best_idx]


class TetrisGame:
    def __init__(self):
        self.env = Tetris()

    def valid_position(self, piece, x, y):
        for py in range(len(piece)):
            for px in range(len(piece[py])):
                if piece[py][px] == 0:
                    continue

                board_x = x + px
                board_y = y + py

                if board_x < 0:
                    return False
                if board_x >= self.env.width:
                    return False
                if board_y >= self.env.height:
                    return False
                if (board_y >= 0 and self.env.board[board_y][board_x]):
                    return False
        return True

    def move_left(self):
        x = self.env.current_pos["x"] - 1
        y = self.env.current_pos["y"]

        if self.valid_position(self.env.piece,x,y):
            self.env.current_pos["x"] -= 1

    def move_right(self):
        x = self.env.current_pos["x"] + 1
        y = self.env.current_pos["y"]

        if self.valid_position(self.env.piece,x,y):
            self.env.current_pos["x"] += 1

    def rotate_piece(self):

        rotated = self.env.rotate(self.env.piece)

        if self.valid_position(rotated,self.env.current_pos["x"],self.env.current_pos["y"]):
            self.env.piece = rotated

    def soft_drop(self):
        x = self.env.current_pos["x"]
        y = self.env.current_pos["y"] + 1

        if self.valid_position(self.env.piece,x,y):
            self.env.current_pos["y"] += 1
            return False
        return True

    def lock_piece(self):
        self.env.board = self.env.store(self.env.piece,self.env.current_pos)
        lines_cleared, self.env.board = (self.env.check_cleared_rows(self.env.board))

        self.env.cleared_lines += lines_cleared
        self.env.tetrominoes += 1
        self.env.new_piece()

    def hard_drop(self):
        while not self.soft_drop():
            pass
        self.lock_piece()

    def AI_moves(self, action):
        target_x, rotations = action
        moves = []
        for _ in range(rotations):
            moves.append("ROTATE")

        current_x = self.env.current_pos["x"]

        while current_x < target_x:
            moves.append("RIGHT")
            current_x += 1

        while current_x > target_x:
            moves.append("LEFT")
            current_x -= 1

        moves.append("DROP")
        return moves
    
    def execute_plan(self, plan):
        for move in plan:
            if move == "LEFT":
                self.move_left()
            elif move == "RIGHT":
                self.move_right()
            elif move == "ROTATE":
                self.rotate_piece()
            elif move == "DROP":
                self.hard_drop()
            self.env.render()

            time.sleep(0.067)

def main():

    game = TetrisGame()

    ai = AIController("final_models/tetris")

    while True:

        action = ai.choose_action(game.env)
        moves = game.AI_moves(action)
        print(moves)

        if action is None:
            break

        game.execute_plan(moves)

        print(
        f"Pieces: {game.env.tetrominoes:5d} | "
        f"Score: {game.env.score:8.2f} | "
        f"Lines: {game.env.cleared_lines:4d} | "
        )


if __name__ == "__main__":
    main()