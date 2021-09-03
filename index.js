require('@tensorflow/tfjs-node'); // calculations on cpu, if on GPU it will 'tfjs-node-gpu'
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv.js');

function knn(features, labels, predictionPoint, k) {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.get(1), 0) / k
  );
}

// shuffle the data to avoid having any kind of biases
// split dataset into test and training datasets
let { features, labels, testFeatures, testLabels } = loadCSV(
  'kc_house_data.csv',
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price'],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);

const err = (testLabels[0][0] - result) / testLabels[0][0];

console.log('Guess', result, testLabels[0][0]);

console.log('Error', err * 100); // -31 % accurate
