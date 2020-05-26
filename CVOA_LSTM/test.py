import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')  # Obtener la lista de GPU's instaladas en la maquina
print("list of visible gpu: {}".format(str(physical_devices)))
