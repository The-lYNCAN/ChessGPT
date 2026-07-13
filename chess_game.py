"""
Chess — a two-player chess game built with pygame + python-chess.

Install dependencies first:
    pip install pygame python-chess

Run:
    python chess_game.py

Controls:
    - Click a piece to select it, then click a highlighted square to move
      (or drag and drop the piece directly).
    - Click a legal destination square (shown with a dot or ring) to move.
    - Promotion: a small picker appears when a pawn reaches the last rank.
    - Buttons at the bottom: Flip board / Undo / New game.
"""

import sys
import json
import chess
import pygame
import torch

pygame.init()
pygame.display.set_caption("Chess")

# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------
# The human always plays White; the model always plays Black. This sidesteps
# the "generate from empty context" problem entirely, since by the time it's
# ever the model's turn, White has already made a move and game_token_ids is
# non-empty.

MODEL_ENABLED = True
MODEL_COLOR = chess.BLACK
HUMAN_COLOR = chess.WHITE
MODEL_TEMPERATURE = 0.5

model = None
token_to_id = None
id_to_token = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_ENABLED:
    try:
        from decoderArchitecture import ChessGPT

        with open("token_to_id", "r") as f:
            token_to_id = json.load(f)
        with open("id_to_token", "r") as f:
            # JSON only supports string keys, so ids were serialized as
            # strings ("0", "1", ...) -- cast back to int on load, or every
            # id_to_token[next_id] lookup below will KeyError.
            id_to_token = {int(k): v for k, v in json.load(f).items()}

        model = ChessGPT(
            vocab_size=len(token_to_id),
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_layers=10,
            max_seq_len=250,
        ).to(device)
        model.load_state_dict(torch.load("checkpoint_epoch_15.pt", map_location=device))
        model.eval()
        print(f"Model loaded on {device}. Black will be played by ChessGPT.")
    except Exception as e:
        print(f"Could not load model, falling back to two-player mode: {e}")
        MODEL_ENABLED = False


def get_model_move(temperature=MODEL_TEMPERATURE):
    """
    Samples one move from the model, constrained to legal moves only.
    Returns a chess.Move, or None if the model has no legal move it
    recognizes in its vocabulary (very rare, but possible with a small vocab).
    """
    legal_uci = [mv.uci() for mv in board.legal_moves]
    legal_ids = [token_to_id[u] for u in legal_uci if u in token_to_id]
    if not legal_ids:
        return None

    input_tensor = torch.tensor([game_token_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_tensor)

    next_logits = logits[0, -1, :] / temperature
    mask = torch.full_like(next_logits, float("-inf"))
    mask[legal_ids] = next_logits[legal_ids]
    probs = torch.softmax(mask, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()

    return chess.Move.from_uci(id_to_token[next_id])

# ---------------------------------------------------------------------------
# Layout / theme
# ---------------------------------------------------------------------------

SQUARE = 74
BOARD_PX = SQUARE * 8
PANEL_W = 320
CONTROLS_H = 60
WIDTH = BOARD_PX + PANEL_W
HEIGHT = BOARD_PX + CONTROLS_H

IVORY = (236, 223, 196)
WALNUT = (124, 90, 58)
WALNUT_DARK = (60, 43, 26)
BG = (23, 20, 15)
PANEL_BG = (34, 29, 22)
BRASS = (201, 162, 75)
BRASS_DIM = (138, 112, 56)
TEXT_PRIMARY = (242, 233, 216)
TEXT_SECONDARY = (180, 165, 138)
TEXT_MUTED = (125, 113, 96)
RED = (181, 69, 61)
SELECT_TINT = (201, 162, 75, 120)
LASTMOVE_TINT = (201, 162, 75, 60)
CHECK_TINT = (181, 69, 61, 130)
DOT_COLOR = (40, 30, 15, 110)

WHITE_PIECE = (236, 223, 196)
BLACK_PIECE = (36, 29, 19)
WHITE_STROKE = (138, 106, 63)
BLACK_STROKE = (201, 162, 75)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

font_title = pygame.font.SysFont("georgia", 26, bold=True)
font_label = pygame.font.SysFont("arial", 12, bold=True)
font_status = pygame.font.SysFont("arial", 16, bold=True)
font_sub = pygame.font.SysFont("arial", 12)
font_move = pygame.font.SysFont("consolas", 14)
font_btn = pygame.font.SysFont("arial", 13, bold=True)
font_coord = pygame.font.SysFont("arial", 10, bold=True)

FILES = "abcdefgh"


def square_name(file_idx, rank_idx):
    return f"{FILES[file_idx]}{rank_idx + 1}"


# ---------------------------------------------------------------------------
# Piece rendering — drawn with primitives, no external image assets
# ---------------------------------------------------------------------------

def scaled(points, cx, cy, size):
    """Map 0..45 design-space points to pixel coordinates centered at (cx, cy)."""
    s = size / 45.0
    off = size / 2.0
    return [(cx - off + x * s, cy - off + y * s) for (x, y) in points]


def draw_piece(surface, piece_type, color, cx, cy, size):
    fill = WHITE_PIECE if color == chess.WHITE else BLACK_PIECE
    stroke = WHITE_STROKE if color == chess.WHITE else BLACK_STROKE
    w = max(1, int(size * 0.03))

    def poly(pts):
        p = scaled(pts, cx, cy, size)
        pygame.draw.polygon(surface, fill, p)
        pygame.draw.polygon(surface, stroke, p, w)

    def circle(x, y, r):
        s = size / 45.0
        off = size / 2.0
        px, py = cx - off + x * s, cy - off + y * s
        pygame.draw.circle(surface, fill, (px, py), r * s)
        pygame.draw.circle(surface, stroke, (px, py), r * s, w)

    def base():
        poly([(13, 34), (32, 34), (32, 39), (13, 39)])

    if piece_type == chess.PAWN:
        circle(22.5, 13, 6)
        poly([(17, 20), (28, 20), (31, 34), (14, 34)])
        base()

    elif piece_type == chess.ROOK:
        poly([(12, 10), (16.5, 10), (16.5, 16), (12, 16)])
        poly([(19, 10), (26, 10), (26, 16), (19, 16)])
        poly([(28.5, 10), (33, 10), (33, 16), (28.5, 16)])
        poly([(12, 16), (33, 16), (33, 20), (12, 20)])
        poly([(14, 20), (31, 20), (29, 32), (16, 32)])
        base()

    elif piece_type == chess.KNIGHT:
        poly([
            (28, 37), (16, 37), (16, 30), (13, 26), (13, 20),
            (17, 13), (22, 11), (21, 8), (25, 9), (26, 6), (29, 9),
            (33, 12), (33, 19), (30, 23), (31, 29), (31, 33),
        ])
        circle(26.5, 15.5, 1.4)
        base()

    elif piece_type == chess.BISHOP:
        circle(22.5, 9, 3)
        poly([(19, 6), (26, 6), (26, 8.5), (19, 8.5)])
        poly([(22.5, 13), (28, 20), (26.5, 27), (29, 30), (22.5, 33), (16, 30), (18.5, 27), (17, 20)])
        poly([(14, 32), (31, 32), (31, 37), (14, 37)])

    elif piece_type == chess.QUEEN:
        circle(12, 11, 2.3)
        circle(17.3, 9, 2.1)
        circle(22.5, 8, 2.5)
        circle(27.7, 9, 2.1)
        circle(33, 11, 2.3)
        poly([(12, 14), (33, 14), (30, 30), (15, 30)])
        base()

    elif piece_type == chess.KING:
        poly([(21, 3), (24, 3), (24, 9), (21, 9)])
        poly([(18, 6), (27, 6), (27, 9), (18, 9)])
        poly([(22.5, 15), (30, 20), (28, 29), (30.5, 32), (22.5, 36), (14.5, 32), (17, 29), (15, 20)])
        base()


def piece_size_for(square, kind):
    return int(SQUARE * 0.82)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

board = chess.Board()
flipped = False
selected_square = None
legal_targets = []
last_move = None
san_history = []
game_token_ids = []  # full move history as token ids, both players, in order
dragging_piece = None
dragging_from = None
drag_pos = (0, 0)
pending_promotion = None  # (from_square, to_square) awaiting piece choice
status_text = "White to move"
status_sub = ""
status_alert = False

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def screen_to_square(px, py):
    if px < 0 or px >= BOARD_PX or py < 0 or py >= BOARD_PX:
        return None
    col = px // SQUARE
    row = py // SQUARE
    file_idx = (7 - col) if flipped else col
    rank_idx = row if flipped else (7 - row)
    return chess.square(file_idx, rank_idx)


def square_to_screen(sq):
    file_idx = chess.square_file(sq)
    rank_idx = chess.square_rank(sq)
    col = (7 - file_idx) if flipped else file_idx
    row = rank_idx if flipped else (7 - rank_idx)
    return col * SQUARE, row * SQUARE


def update_status():
    global status_text, status_sub, status_alert
    status_alert = False
    turn_name = "White" if board.turn == chess.WHITE else "Black"

    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None:
            status_text = "Draw"
            reason = outcome.termination
            status_sub = str(reason).split(".")[-1].replace("_", " ").title()
        else:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            status_text = f"{winner} wins"
            status_sub = "Checkmate"
            status_alert = True
    else:
        status_text = f"{turn_name} to move"
        status_sub = "Check" if board.is_check() else ""
        status_alert = board.is_check()


def rebuild_san_history():
    global san_history
    san_history = []
    temp = chess.Board()
    for mv in board.move_stack:
        san_history.append(temp.san(mv))
        temp.push(mv)


def make_move(from_sq, to_sq, promotion=None):
    global last_move
    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move not in board.legal_moves:
        # try to find a matching legal move (handles promotion default cases)
        candidates = [m for m in board.legal_moves if m.from_square == from_sq and m.to_square == to_sq]
        if not candidates:
            return False
        move = candidates[0]

    uci = move.uci()
    san_history.append(board.san(move))
    board.push(move)
    last_move = move
    if MODEL_ENABLED and token_to_id is not None:
        game_token_ids.append(token_to_id.get(uci, token_to_id.get("<UNK>", 0)))
    update_status()

    if MODEL_ENABLED and board.turn == MODEL_COLOR and not board.is_game_over():
        take_model_turn()

    return True


def take_model_turn():
    """Shows a 'thinking' status, runs one forward pass, and plays the result."""
    global status_text, status_sub
    status_text = "Black is thinking..."
    status_sub = ""
    render()  # paint the "thinking" state before the (blocking) model call

    model_move = get_model_move()
    if model_move is None:
        status_text = "Model has no legal move it recognizes"
        status_sub = "Game paused"
        return

    make_move(model_move.from_square, model_move.to_square, promotion=model_move.promotion)


def is_promotion_move(from_sq, to_sq):
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    to_rank = chess.square_rank(to_sq)
    return (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0)


def attempt_move(from_sq, to_sq):
    global selected_square, legal_targets, pending_promotion
    if from_sq == to_sq:
        selected_square = None
        legal_targets = []
        return

    if is_promotion_move(from_sq, to_sq):
        pending_promotion = (from_sq, to_sq)
        selected_square = None
        legal_targets = []
        return

    made = make_move(from_sq, to_sq)
    selected_square = None
    legal_targets = []
    if not made:
        piece = board.piece_at(to_sq)
        if piece and piece.color == board.turn:
            select_square(to_sq)


def select_square(sq):
    global selected_square, legal_targets
    piece = board.piece_at(sq)
    if piece is None or piece.color != board.turn:
        selected_square = None
        legal_targets = []
        return
    selected_square = sq
    legal_targets = [m for m in board.legal_moves if m.from_square == sq]


def new_game():
    global board, selected_square, legal_targets, last_move, san_history, pending_promotion, game_token_ids
    board = chess.Board()
    selected_square = None
    legal_targets = []
    last_move = None
    san_history = []
    game_token_ids = []
    pending_promotion = None
    update_status()


def undo_move():
    global selected_square, legal_targets, last_move, pending_promotion

    def pop_one():
        board.pop()
        if san_history:
            san_history.pop()
        if game_token_ids:
            game_token_ids.pop()

    if board.move_stack:
        pop_one()
        # if we're now sitting right before the model's turn, that means we
        # only undid the model's reply -- undo the human's move too, so
        # control lands back with the human instead of stalling on the model
        if MODEL_ENABLED and board.move_stack and board.turn == MODEL_COLOR:
            pop_one()

        last_move = board.peek() if board.move_stack else None
        selected_square = None
        legal_targets = []
        pending_promotion = None
        update_status()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_board():
    for row in range(8):
        for col in range(8):
            file_idx = (7 - col) if flipped else col
            rank_idx = row if flipped else (7 - row)
            sq = chess.square(file_idx, rank_idx)
            is_light = (file_idx + rank_idx) % 2 == 1
            color = IVORY if is_light else WALNUT
            x, y = col * SQUARE, row * SQUARE
            pygame.draw.rect(screen, color, (x, y, SQUARE, SQUARE))

            if last_move is not None and sq in (last_move.from_square, last_move.to_square):
                overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                overlay.fill(LASTMOVE_TINT)
                screen.blit(overlay, (x, y))

            if selected_square == sq:
                overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                overlay.fill(SELECT_TINT)
                screen.blit(overlay, (x, y))

            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.KING and piece.color == board.turn and board.is_check():
                overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                overlay.fill(CHECK_TINT)
                screen.blit(overlay, (x, y))

            if col == (7 if flipped else 0):
                label = font_coord.render(str(rank_idx + 1), True, WALNUT_DARK if is_light else IVORY)
                screen.blit(label, (x + 4, y + 3))
            if row == (0 if flipped else 7):
                label = font_coord.render(FILES[file_idx], True, WALNUT_DARK if is_light else IVORY)
                screen.blit(label, (x + SQUARE - 12, y + SQUARE - 16))

            for mv in legal_targets:
                if mv.to_square == sq:
                    center = (x + SQUARE // 2, y + SQUARE // 2)
                    if board.piece_at(sq) is not None or (board.is_en_passant(mv)):
                        s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                        pygame.draw.circle(s, DOT_COLOR, (SQUARE // 2, SQUARE // 2), SQUARE // 2 - 5, 4)
                        screen.blit(s, (x, y))
                    else:
                        s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                        pygame.draw.circle(s, DOT_COLOR, (SQUARE // 2, SQUARE // 2), 9)
                        screen.blit(s, (x, y))


def draw_pieces():
    for sq in chess.SQUARES:
        if dragging_from == sq:
            continue
        piece = board.piece_at(sq)
        if piece is None:
            continue
        x, y = square_to_screen(sq)
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        draw_piece(screen, piece.piece_type, piece.color, cx, cy, int(SQUARE * 0.8))

    if dragging_piece is not None:
        draw_piece(screen, dragging_piece.piece_type, dragging_piece.color, drag_pos[0], drag_pos[1], int(SQUARE * 0.85))


def draw_panel():
    px = BOARD_PX
    pygame.draw.rect(screen, PANEL_BG, (px, 0, PANEL_W, BOARD_PX))

    title = font_title.render("Chess", True, IVORY)
    screen.blit(title, (px + 20, 18))
    pygame.draw.line(screen, BRASS_DIM, (px + 20, 56), (px + 90, 56), 1)

    dot_color = IVORY if board.turn == chess.WHITE else BLACK_PIECE
    pygame.draw.circle(screen, dot_color, (px + 30, 84), 7)
    pygame.draw.circle(screen, BRASS_DIM, (px + 30, 84), 7, 1)

    status_color = RED if status_alert else TEXT_PRIMARY
    st = font_status.render(status_text, True, status_color)
    screen.blit(st, (px + 46, 76))
    if status_sub:
        sub = font_sub.render(status_sub, True, RED if status_alert else TEXT_SECONDARY)
        screen.blit(sub, (px + 46, 94))

    y = 130
    label = font_label.render("WHITE CAPTURED", True, TEXT_MUTED)
    screen.blit(label, (px + 20, y))
    taken_by_white = [m.split("x")[0] for m in []]  # placeholder, replaced below
    draw_captured(px + 20, y + 18, chess.WHITE)

    y2 = 210
    label2 = font_label.render("BLACK CAPTURED", True, TEXT_MUTED)
    screen.blit(label2, (px + 20, y2))
    draw_captured(px + 20, y2 + 18, chess.BLACK)

    y3 = 290
    label3 = font_label.render("MOVE HISTORY", True, TEXT_MUTED)
    screen.blit(label3, (px + 20, y3))
    draw_history(px + 20, y3 + 20)

    draw_buttons()


def captured_pieces():
    """Return (captured_by_white, captured_by_black) as lists of piece_type."""
    temp = chess.Board()
    by_white, by_black = [], []
    for mv in board.move_stack:
        if temp.is_capture(mv):
            captured_piece = temp.piece_at(mv.to_square)
            if temp.is_en_passant(mv):
                captured_type = chess.PAWN
            else:
                captured_type = captured_piece.piece_type if captured_piece else chess.PAWN
            if temp.turn == chess.WHITE:
                by_white.append(captured_type)
            else:
                by_black.append(captured_type)
        temp.push(mv)
    return by_white, by_black


def draw_captured(x, y, side):
    by_white, by_black = captured_pieces()
    pieces = by_white if side == chess.WHITE else by_black
    opponent_color = chess.BLACK if side == chess.WHITE else chess.WHITE

    cx = x
    for pt in pieces:
        draw_piece(screen, pt, opponent_color, cx + 10, y + 10, 22)
        cx += 20
        if cx > x + PANEL_W - 70:
            break

    score = sum(VALUES.get(pt, 0) for pt in by_white)
    other = sum(VALUES.get(pt, 0) for pt in by_black)
    diff = score - other if side == chess.WHITE else other - score
    if diff > 0:
        diff_label = font_sub.render(f"+{diff}", True, BRASS)
        screen.blit(diff_label, (x + PANEL_W - 60, y + 4))


def draw_history(x, y):
    row_h = 18
    max_rows = 9
    start = max(0, len(san_history) - max_rows * 2)
    visible = san_history[start:]

    if not san_history:
        empty = font_move.render("No moves yet", True, TEXT_MUTED)
        screen.blit(empty, (x, y))
        return

    first_move_num = start // 2 + 1
    row_y = y
    i = 0
    while i < len(visible):
        num = first_move_num + i // 2
        white_move = visible[i] if i < len(visible) else ""
        black_move = visible[i + 1] if i + 1 < len(visible) else ""
        num_label = font_move.render(f"{num}.", True, TEXT_MUTED)
        w_label = font_move.render(white_move, True, TEXT_PRIMARY)
        b_label = font_move.render(black_move, True, TEXT_PRIMARY)
        screen.blit(num_label, (x, row_y))
        screen.blit(w_label, (x + 28, row_y))
        screen.blit(b_label, (x + 110, row_y))
        row_y += row_h
        i += 2


BUTTONS = []


def draw_buttons():
    global BUTTONS
    BUTTONS = []
    labels = ["Flip board", "Undo move", "New game"]

    # buttons live in the bottom control strip across the full window width
    n = len(labels)
    total_w = WIDTH - 40
    gap = 12
    each_w = (total_w - gap * (n - 1)) // n
    y = BOARD_PX + (CONTROLS_H - 34) // 2
    x = 20
    for label in labels:
        rect = pygame.Rect(x, y, each_w, 34)
        pygame.draw.rect(screen, BG, rect, border_radius=8)
        pygame.draw.rect(screen, BRASS_DIM, rect, 1, border_radius=8)
        text = font_btn.render(label, True, TEXT_PRIMARY)
        tx = rect.x + (rect.width - text.get_width()) // 2
        ty = rect.y + (rect.height - text.get_height()) // 2
        screen.blit(text, (tx, ty))
        BUTTONS.append((label, rect))
        x += each_w + gap


def draw_promotion_picker():
    if pending_promotion is None:
        return
    from_sq, to_sq = pending_promotion
    piece = board.piece_at(from_sq)
    color = piece.color

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((10, 8, 5, 160))
    screen.blit(overlay, (0, 0))

    choices = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    card_w = 4 * 64 + 5 * 12
    card_h = 88
    card_x = (WIDTH - card_w) // 2
    card_y = (HEIGHT - card_h) // 2
    pygame.draw.rect(screen, PANEL_BG, (card_x, card_y, card_w, card_h), border_radius=12)
    pygame.draw.rect(screen, BRASS_DIM, (card_x, card_y, card_w, card_h), 1, border_radius=12)

    rects = []
    x = card_x + 12
    for pt in choices:
        rect = pygame.Rect(x, card_y + 12, 64, 64)
        pygame.draw.rect(screen, BG, rect, border_radius=10)
        pygame.draw.rect(screen, BRASS_DIM, rect, 1, border_radius=10)
        draw_piece(screen, pt, color, rect.centerx, rect.centery, 44)
        rects.append((pt, rect))
        x += 64 + 12

    return rects


promotion_rects = []


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def handle_click(pos):
    global selected_square, dragging_piece, dragging_from

    for label, rect in BUTTONS:
        if rect.collidepoint(pos):
            if label == "Flip board":
                toggle_flip()
            elif label == "Undo move":
                undo_move()
            elif label == "New game":
                new_game()
            return

    sq = screen_to_square(*pos)
    if sq is None:
        return

    if selected_square is not None:
        if any(mv.to_square == sq for mv in legal_targets):
            attempt_move(selected_square, sq)
            return
        piece = board.piece_at(sq)
        if piece and piece.color == board.turn:
            select_square(sq)
        else:
            selected_square = None
    else:
        select_square(sq)


def toggle_flip():
    global flipped
    flipped = not flipped


def handle_promotion_click(pos, rects):
    global pending_promotion
    for pt, rect in rects:
        if rect.collidepoint(pos):
            from_sq, to_sq = pending_promotion
            pending_promotion = None
            make_move(from_sq, to_sq, promotion=pt)
            return


def render():
    screen.fill(BG)
    draw_board()
    draw_pieces()
    draw_panel()
    pygame.display.flip()


def main():
    global dragging_piece, dragging_from, drag_pos, selected_square, promotion_rects

    update_status()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pending_promotion is not None:
                    handle_promotion_click(event.pos, promotion_rects)
                    continue

                px, py = event.pos
                sq = screen_to_square(px, py)
                clicked_button = any(rect.collidepoint(event.pos) for _, rect in BUTTONS)
                if sq is not None and not clicked_button:
                    piece = board.piece_at(sq)
                    if piece and piece.color == board.turn:
                        dragging_from = sq
                        dragging_piece = piece
                        drag_pos = event.pos
                        select_square(sq)
                        continue
                handle_click(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                if dragging_from is not None:
                    drag_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if dragging_from is not None:
                    target = screen_to_square(*event.pos)
                    from_sq = dragging_from
                    dragging_from = None
                    dragging_piece = None
                    if target is not None:
                        if any(mv.to_square == target for mv in legal_targets):
                            attempt_move(from_sq, target)
                        else:
                            selected_square = None
                    else:
                        selected_square = None

        screen.fill(BG)
        draw_board()
        draw_pieces()
        draw_panel()

        if pending_promotion is not None:
            promotion_rects = draw_promotion_picker()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()