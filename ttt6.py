import numpy as np
import random
import sys

# ---------- Загрузка модели ----------
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Ошибка: TensorFlow не установлен.")
    print("Установите: pip install tensorflow")
    sys.exit(1)

MODEL_PATH = 'tic_tac_toe_model_improved.keras'   # ← изменил на новый формат

MODEL_PATH = "tic_tac_toe_dqn_selfplay2.keras"

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Модель успешно загружена: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Не удалось загрузить модель: {e}")
    print("Убедитесь, что файл модели существует в папке со скриптом.")
    sys.exit(1)


# ---------- Вспомогательные функции ----------
def board_to_input(board):
    """Преобразование доски в формат модели (27 признаков)"""
    x = np.zeros((9, 3), dtype=np.float32)
    for i in range(9):
        if board[i] == 1:
            x[i, 0] = 1.0
        elif board[i] == -1:
            x[i, 1] = 1.0
        else:
            x[i, 2] = 1.0
    return x.flatten()


def print_board(board):
    print("\n   A   B   C")
    for row in range(3):
        line = f"{row+1}  "
        for col in range(3):
            idx = row * 3 + col
            val = board[idx]
            if val == 1:
                line += " X "
            elif val == -1:
                line += " O "
            else:
                line += " . "
            if col < 2:
                line += "│"
        print(line)
        if row < 2:
            print("   ───┼───┼───")
    print()


def check_winner(board):
    win_patterns = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a, b, c in win_patterns:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    if 0 not in board:
        return 0
    return None


def user_move(board):
    while True:
        move_str = input("Ваш ход (например A1, B3, C2): ").strip().upper()
        if len(move_str) != 2:
            print("❌ Некорректный формат. Пример: A1")
            continue
        
        col_letter, row_digit = move_str[0], move_str[1]
        if col_letter not in 'ABC' or row_digit not in '123':
            print("❌ Используйте буквы A-C и цифры 1-3")
            continue
        
        col = ord(col_letter) - ord('A')
        row = int(row_digit) - 1
        idx = row * 3 + col
        
        if board[idx] != 0:
            print("❌ Клетка уже занята!")
            continue
            
        return idx


def predict_best_move(board):
    """Предсказание лучшего хода с помощью модели"""
    legal_moves = [i for i in range(9) if board[i] == 0]
    
    if not legal_moves:
        return None
    
    input_data = np.array([board_to_input(board)], dtype=np.float32)
    preds = model.predict(input_data, verbose=0)[0]
    
    # Маскируем запрещённые ходы
    for i in range(9):
        if i not in legal_moves:
            preds[i] = -np.inf
    
    best_move = int(np.argmax(preds))
    return best_move


# ---------- Основная игра ----------
def play_game():
    board = [0] * 9
    first = random.choice([0, 1])  # 0 - человек первый
    
    if first == 0:
        print("🎮 Вы ходите первым! Вы играете за ❌ (крестики)")
        human_symbol = 1
        computer_symbol = -1
    else:
        print("🤖 Компьютер ходит первым! Вы играете за ⭕ (нолики)")
        human_symbol = -1
        computer_symbol = 1
    
    turn = 0 if first == 0 else 1  # 0 - человек, 1 - компьютер
    
    while True:
        print_board(board)
        
        if turn == 0:  # Ход человека
            print(f"Ваш ход ({'❌' if human_symbol == 1 else '⭕'})")
            idx = user_move(board)
            board[idx] = human_symbol
        else:  # Ход компьютера
            print("Ход компьютера...")
            idx = predict_best_move(board)
            board[idx] = computer_symbol
            print(f"Компьютер поставил в {chr(ord('A') + idx % 3)}{idx // 3 + 1}")
        
        winner = check_winner(board)
        
        if winner is not None:
            print_board(board)
            if winner == 1:
                print("🏆 Победили крестики (X)!")
            elif winner == -1:
                print("🏆 Победили нолики (O)!")
            else:
                print("🤝 Ничья!")
            break
        
        turn = 1 - turn


if __name__ == "__main__":
    print("🎲 Игра в крестики-нолики с ИИ (улучшенная модель)\n")
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\nИгра прервана.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        input("\nНажмите Enter для выхода...")