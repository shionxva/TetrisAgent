import cv2

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
        self.move_queue = []
        self.ai_enabled = True

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
        score_table = {
            0: 0,
            1: 100,
            2: 300,
            3: 500,
            4: 800
        }
        self.env.score += score_table.get(lines_cleared,0)
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
    
    def Human_moves(self, key):
        LEFT_ARROW  = 2424832
        UP_ARROW    = 2490368
        RIGHT_ARROW = 2555904
        DOWN_ARROW  = 2621440

        if key == LEFT_ARROW:
            self.move_left()
        elif key == RIGHT_ARROW:
            self.move_right()
        elif key == UP_ARROW:
            self.rotate_piece()
        elif key == DOWN_ARROW:
            self.soft_drop()
        elif key == ord(' '):
            self.hard_drop()
            return True
        return False
    
    def format_moves(self, moves):
        if not moves:
            return ""

        mapping = {
            "ROTATE": "Rot",
            "LEFT": "L",
            "RIGHT": "R",
            "DROP": "Drop"
        }

        result = []
        current = moves[0]
        count = 1

        for move in moves[1:]:
            if move == current:
                count += 1
            else:
                text = mapping[current]
                if count > 1:
                    result.append(f"{text}{count}")
                else:
                    result.append(text)
                current = move
                count = 1

        text = mapping[current]
        if count > 1:
            result.append(f"{text}{count}")
        else:
            result.append(text)

        return " ".join(result)
    
    # def execute_moves(self, moves):
    #     for move in moves:
    #         self.env.ai_current_move = f"> {move}" # Update the AI move display
    #         if move == "LEFT":
    #             self.move_left()
    #         elif move == "RIGHT":
    #             self.move_right()
    #         elif move == "ROTATE":
    #             self.rotate_piece()
    #         elif move == "DROP":
    #             self.hard_drop()
    #         self.env.render()

    #         time.sleep(0.067)

    def execute_next_move(self):
        if not self.move_queue:
            return False

        move = self.move_queue.pop(0)
        self.env.ai_current_move = move

        if move == "LEFT":
            self.move_left()

        elif move == "RIGHT":
            self.move_right()

        elif move == "ROTATE":
            self.rotate_piece()

        elif move == "DROP":
            self.hard_drop()
            return True   # piece locked
        return False

def main():

    game = TetrisGame()
    ai = AIController("final_models/tetris")

    gravity_delay = 0.30
    ai_delay = 0.05

    last_gravity = time.time()
    last_ai = time.time()

    moves_seq = None

    while not game.env.gameover:

        game.env.ai_status = game.ai_enabled

        key = cv2.waitKeyEx(1)

        # ======================
        # Toggle AI
        # ======================
        if key == ord('t'):
            game.ai_enabled = not game.ai_enabled

            print(f"AI {'ON' if game.ai_enabled else 'OFF'}")

        # ======================
        # Human Controls
        # ======================
        if not game.ai_enabled:

            dropped = game.Human_moves(key)

            if dropped:
                game.move_queue.clear()
                moves_seq = None

        # ======================
        # AI Prediction
        # ======================
        moves_seq = ai.choose_action(game.env)

        if moves_seq is not None:

            predicted_moves = game.AI_moves(moves_seq)

            game.env.ai_moves = game.format_moves(
                predicted_moves
            )

        now = time.time()

        # ======================
        # Gravity
        # ======================
        if now - last_gravity >= gravity_delay:

            should_lock = game.soft_drop()

            if should_lock:

                game.lock_piece()

                game.move_queue.clear()
                moves_seq = None

                print(
                    f"Pieces: {game.env.tetrominoes:5d} | "
                    f"Score: {game.env.score:8.2f} | "
                    f"Lines: {game.env.cleared_lines:4d}"
                )

            last_gravity = now

        # ======================
        # AI Execution
        # ======================
        if game.ai_enabled:
            if now - last_ai >= ai_delay:
                if moves_seq is not None:
                    game.move_queue = game.AI_moves(moves_seq)
                    game.execute_next_move()
                last_ai = now
        game.env.render()
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()