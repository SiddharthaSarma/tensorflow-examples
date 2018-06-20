const Iris = (function() {
  const init = async () => {
    const { irisRawData, testData } = await loadIrisData();
    doIrisClassification(irisRawData, testData);
  };

  const loadIrisData = async () => {
    const rawData = await fetch('data/iris-training.json');
    const rawTestData = await fetch('data/iris.json');
    const irisRawData = await rawData.json();
    const testData = await rawTestData.json();
    return {
      irisRawData,
      testData
    };
  };

  function doIrisClassification(rawTrainingData, testData) {
    console.log(rawTrainingData, testData);
    // convert/setup our data
    const trainingdata = tf.tensor2d(
      rawTrainingData.map(item => [
        item.sepallength,
        item.sepalwidth,
        item.petallength,
        item.petalwidth
      ])
    );
    const testingData = tf.tensor2d(
      testData.map(item => [
        item.sepallength,
        item.sepalwidth,
        item.petallength,
        item.petalwidth
      ])
    );
    const outputData = tf.tensor2d(
      rawTrainingData.map(item => [
        item.species === 'setosa' ? 1 : 0,
        item.species === 'virginica' ? 1 : 0,
        item.species === 'versicolor' ? 1 : 0
      ])
    );
    outputData.print();

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

    // train
    model.fit(trainingdata, outputData, { epochs: 1000 }).then(history => {
      model.predict(testingData).print();
    });
  }

  return {
    init
  };
})();

Iris.init();
