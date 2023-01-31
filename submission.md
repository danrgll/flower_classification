##### Which track(s) are you participating in? {Fast Networks Track, Large Networks Track, Both}
Both

##### What are the number of learnable parameters in your model?
Fast Networks Track - 99,346
Large Networks Track - 99,346

##### Briefly describe your approach
Fast Networks Track - Try different models to overfit the data. Then try fight against overfitting with 
different Regularization Techniques. Starting with Augmentations of the Data, add Dropout, BatchNorm and weight decay.
We played arround with different optimizers(sgd, adam, adamw) and Learning rates scheduler(Cosine Annealing/with and 
without Warm Restart to increase our performance. We tuned Hyperparameters manually and automated (lr_rate_finder, 
bayesian, hyperband). We observed with monitoring how this changes affected our models, and used the promesing ones to
come to our final model:
- Model architecture: ModelZeroSeven (in cnn.py)
- Augmentations: TrivialAugment, add additionally with a 5% chance Gaussian Blur or Contrast on each image
- Optimizer: SGD with Weight Decay
- Learning rate scheduler: CosineAnnealing with Warm Restart
- Epochs 1270

Large Networks Track - same net_architecture and hyperparameters but different model, thats only trained on Training 
set (not on val or test).
Through training we save the best working model on validation and get a model that works quite well on val and test.
The parameters get saved through this process in src as "best_model_parameters.pt".
We would like to see how this performance on the hidden data set, with just learned on the train data set

##### Command to train your model from scratch
Fast Networks Track - python -m src.main --model ModelZeroSeven --epochs 1270 --data-augmentation trivial_augment --use-all-data-to-train
Large Networks Track - python -m src.main --model ModelZeroSeven --epochs 1270 --data-augmentation trivial_augment

##### Command to evaluate your model
Fast Networks Track - python -m src.evaluate_model --model ModelZeroSeven --saved-model-file fast_model
Large Networks Track - python -m src.evaluate_model --model ModelZeroSeven --saved-model-file large_model
