# Comparison: the same model in PyTorch and Keras

I built the same network in both frameworks: 8 inputs, two hidden layers
of 32 units with ReLU, one output, 1377 parameters. After copying the
PyTorch weights into Keras, the largest gap between predictions on the
same rows was 3e-08, so before training both are the same function. Both
versions used the same rows, Adam, learning rate 1e-3, batch size 64, MSE
loss and 100 epochs, on seeds 42, 43 and 44.

## What I observed
1. **Accuracy.** Keras had the lower validation MAE on all three seeds
   (mean 4.02, against 5.26 for PyTorch) and the lower test MAE (3.94
   against 5.16). The likely reason is the different starting weights, not
   the framework. All runs were still improving at epoch 100, and three
   seeds are too few to prove the cause.
2. **Time.** PyTorch was faster: 12.5 seconds per run on average, against
   18.1 for Keras.
3. **Code.** The PyTorch loop (`run_epoch` and `fit`) is about 35 lines
   that I had to get right, including putting back the best weights by
   hand. In Keras it is one `fit` call, and `restore_best_weights=True`
   does the same job.
4. **Saving.** My PyTorch code saves no model. Keras saved a 41 KB
   checkpoint through `ModelCheckpoint` with no extra code.
5. **Errors.** When I copied the weights without the transpose, Keras
   raised a ValueError on the first and last layers. The middle layer
   (32 by 32) would have passed silently, so the equivalence test uses
   different random weights.

## Recommendation
I recommend PyTorch for Lakeshore Bike Share.
- The team may try Hugging Face models later. Hugging Face has said its
  Transformers library (version 5) is making PyTorch its only backend and
  sunsetting TensorFlow support, so PyTorch is the safer path.
- The weekly job already works and is tested in PyTorch, so a small team
  keeps one code base.
- PyTorch was faster in my runs.

## What the team gives up
They write and maintain the training loop by hand, and get no built-in
checkpoint or early stopping. The team member who knows Keras must learn
PyTorch.

## What would change my mind
If the team had to deploy the model with TensorFlow tools (for example
TensorFlow Lite on phones), or if ten or more seeds showed that Keras is
clearly more accurate, I would choose Keras.