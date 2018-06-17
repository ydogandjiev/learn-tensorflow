import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-node";

function exampleOne() {
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // Generate some synthetic data for training.
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  // Train the model using the data.
  return model.fit(xs, ys, { epochs: 10 }).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([5], [1, 1])).print();
  });
}

function exampleTwo() {
  // Train a simple model:
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 100, activation: "relu", inputShape: [10] })
  );
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  const xs = tf.randomNormal([100, 10]);
  const ys = tf.randomNormal([100, 1]);

  return model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        console.log(`Epoch ${epoch}: loss = ${log.loss}`);
      }
    }
  });
}

if (Date.now() % 2) {
  exampleOne();
} else {
  exampleTwo();
}
