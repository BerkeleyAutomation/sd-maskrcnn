import os, numpy as np
import maskrcnn.model as modellib
from maskrcnn.model import log
from clutter import ClutterDataset, ClutterConfig
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

def main(_):
  # Load the datasets, configs.
  config = ClutterConfig()
  config.display()
  model_dir = os.path.join('outputs/v1/') 

  # Training dataset
  dataset_train = ClutterDataset()
  dataset_train.load('train', 'gray', 0)
  dataset_train.prepare()

  # Validation dataset
  dataset_val = ClutterDataset()
  dataset_val.load('test', 'gray', 0)
  dataset_val.prepare()

  # Create the model.
  model = modellib.MaskRCNN(mode="training", config=config,
    model_dir=model_dir)
  
  # model.load_weights(model.find_last()[1], by_name=True)

  # Train the model.
  # Train the head branches
  # Passing layers="heads" freezes all layers except the head
  # layers. You can also pass a regular expression to select
  # which layers to train by name pattern.
  model.train(dataset_train, dataset_val, 
              learning_rate=config.LEARNING_RATE, 
              epochs=10, 
              layers='all')

if __name__ == '__main__':
  app.run()
