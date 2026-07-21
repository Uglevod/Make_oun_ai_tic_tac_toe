import keras
import tf2onnx
import tensorflow as tf

# 1. Загружаем сохраненную Keras-модель
model = keras.saving.load_model('tic_tac_toe_dqn_selfplay2.keras')

# 2. Определяем входной формат (у вас на вход идет вектор из 27 чисел)
# None означает, что размер батча может быть любым (например, 1 для одиночного хода)
input_signature = [tf.TensorSpec([None, 27], tf.float32, name="input_board")]

# 3. Конвертируем и сохраняем в файл .onnx
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, output_path="tic_tac_toe.onnx")
print("✅ Модель успешно конвертирована в 'tic_tac_toe.onnx'!")
