const Iris = (function() {

  const init = async () => {
    const {
      irisRawData,
      testData
    } = await loadIrisData();
    doIrisClassification(irisRawData, testData);
  }

  const loadIrisData = async () => {
    const [rawData, rawTestData] = await Promise.all([fetch('data/iris-training.json'), fetch('data/iris.json')]);
    const [irisRawData, testData] = await Promise.all([rawData.json(), rawTestData.json()]);
    return {
      irisRawData,
      testData
    };
  };

  function doIrisClassification(rawTrainingData, testData) {
    // convert/setup our data
    const trainingData = tf.tensor2d(
      rawTrainingData.map(item => [
        item.sepalLength,
        item.sepalWidth,
        item.petalLength,
        item.petalWidth
      ])
    );
    const testingData = tf.tensor2d(
      testData.map(item => [
        item.sepalLength,
        item.sepalWidth,
        item.petalLength,
        item.petalWidth
      ])
    );
    const outputData = tf.tensor2d(
      rawTrainingData.map(item => [
        item.species === 'setosa' ? 1 : 0,
        item.species === 'virginica' ? 1 : 0,
        item.species === 'versicolor' ? 1 : 0
      ])
    );

    // build nueral network
    const model = tf.sequential();

    model.add(
      tf.layers.dense({
        inputShape: [4],
        activation: 'sigmoid',
        units: 5
      })
    );
    model.add(
      tf.layers.dense({
        inputShape: [5],
        activation: 'sigmoid',
        units: 3
      })
    );
    model.add(
      tf.layers.dense({
        activation: 'sigmoid',
        units: 3
      })
    );

    model.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.adam(0.06)
    });


    const time = Date.now();
    // train
    model.fit(trainingData, outputData, {
      epochs: 100
    }).then(history => {
      console.log(`%c Time taken to train is ${Date.now() - time} seconds`, `background: #222; color: #bada55`)
      model.predict(testingData).print();
    });
  }

  return {
    init
  };

})();

Iris.init();
