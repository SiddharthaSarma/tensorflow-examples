const Iris = (function() {
  function init() {
    loadIrisData();
  }

  async loadIrisData() {
    const rawData = await fetch('../iris.json');
    const data = await rawData.json();
    console.log(data);
  }

  return {
    init
  };
})();

Iris.init();