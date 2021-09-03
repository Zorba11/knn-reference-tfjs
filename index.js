require('@tensorflow/tfjs-node'); // calculations on cpu, if on GPU it will 'tfjs-node-gpu'
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv.js');

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPrediction)
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
    dataColumns: ['lat', 'long', 'sqft_lot'],
    labelColumns: ['price'],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);

  // testLabels[i][0] -> is the testLabel (or price) for the corresponding testPoint (or lat|long)
  // console.log(testPoint);
  // console.log('testLabel..', testLabels, i, testLabels[i][0]);

  const err = (testLabels[i][0] - result) / testLabels[i][0];

  console.log('Guess', result, testLabels[i][0]);

  console.log('Error', err * 100);
});
