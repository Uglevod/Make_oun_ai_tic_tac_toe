import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ====================== КОНФИГУРАЦИЯ ======================
EPISODES = 2500        # Уменьшено, так как игра простая
BATCH_SIZE = 64
MEMORY_SIZE = 10000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995     # Затухание будет применяться каждый эпизод

# ====================== DQN АГЕНТ ======================
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Упрощенная архитектура. Для 27 входов и 9 выходов 128 нейронов более чем достаточно
        model = Sequential([
            Dense(128, activation='relu', input_shape=(27,)),
            Dense(128, activation='relu'),
            Dense(9, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def board_to_input(self, board, current_player):
        """
        Представляем доску ОТНОСИТЕЛЬНО текущего игрока.
        Канал 0: Мои фигуры
        Канал 1: Фигуры противника
        Канал 2: Пустые клетки
        """
        x = np.zeros((9, 3), dtype=np.float32)
        for i in range(9):
            if board[i] == current_player:
                x[i, 0] = 1.0  # Моя фигура
            elif board[i] == -current_player:
                x[i, 1] = 1.0  # Фигура врага
            else:
                x[i, 2] = 1.0  # Пусто
        return x.flatten()

    def act(self, board, legal_moves, current_player):
        # Исследование
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
        
        state = self.board_to_input(board, current_player).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        
        # Маскируем нелегальные ходы
        q_values_masked = np.full(9, -np.inf)
        for move in legal_moves:
            q_values_masked[move] = q_values[move]
        
        return int(np.argmax(q_values_masked))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        states = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + GAMMA * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)

# ====================== ИГРА ======================
def check_winner(board):
    patterns = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in patterns:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0 if 0 not in board else None

def get_legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

# ====================== ОБУЧЕНИЕ ======================
agent = DQNAgent()
print("Начинаем самообучение Tic-Tac-Toe...")

for e in range(EPISODES):
    board = [0] * 9
    player = 1  # 1 = Игрок 1, -1 = Игрок 2
    
    # Сохраняем историю ходов для правильной раздачи наград в конце
    history = [] 

    while True:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            break
            
        # Относительное представление доски для текущего игрока
        state = agent.board_to_input(board, player)
        action = agent.act(board, legal_moves, player)
        
        # Сохраняем ход в историю (состояние, действие, кто ходил)
        history.append((state, action, player))
        
        board[action] = player
        winner = check_winner(board)
        
        if winner is not None:
            break # Игра окончена
            
        player = -player # Смена хода
    
    # === ПРАВИЛЬНАЯ РАЗДАЧА НАГРАД ПОСЛЕ ОКОНЧАНИЯ ПАРТИИ ===
    done = True
    for i, (state, action, p) in enumerate(history):
        # Состояние после хода (для промежуточных шагов оно не важно, так как reward=0)
        next_state = agent.board_to_input(board, -p) 
        
        if winner == 0:
            reward = 0.0  # Ничья
        elif winner == p:
            reward = 1.0  # Победа
        else:
            reward = -1.0 # Поражение
            
        agent.remember(state, action, reward, next_state, done)

    # Обучение на буфере памяти
    agent.replay()
    
    # Обновление целевой сети и затухание epsilon КАЖДЫЙ эпизод
    if e % 50 == 0:
        agent.update_target_model()
        
    agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
    
    if e % 500 == 0:
        print(f"Эпизод {e}/{EPISODES} | Epsilon: {agent.epsilon:.4f} | Memory: {len(agent.memory)}")

print("✅ Обучение завершено!")
agent.model.save('tic_tac_toe_dqn_selfplay2.keras')