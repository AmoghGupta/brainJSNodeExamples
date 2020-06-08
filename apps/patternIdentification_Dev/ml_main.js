const brain = require('brain.js');
const fs = require('fs');
// learning model file
const data = require('./data.json');

const networkPath = 'solutions-cached.network.json';

//training data for our machine learning algo to learn
const trainingData = data.map(item => ({
  input: item.text,
  output: item.category
}));

// telling our computer what we want to do
// LSTM IN RNN
const network = new brain.recurrent.LSTM();

let networkData = null;

// logic is if you already have a dataset which you already trained
// go through network path
//if your learning data set is updated, delete n/w path and run this code
// so that it creates a new n/w path with the updated learning data set

//NETWORK PATH: quick recall
if (fs.existsSync(networkPath)) {
  networkData = JSON.parse(fs.readFileSync(networkPath));
  network.fromJSON(networkData);
} else {
  //train the network with your model
  // network.train(trainingData, {
  //   iterations: 2000, /// default is 20k, its the maximum times to iterate the training data
  //   errorThresh: 0.005, // the acceptable error percentage from training data --> number between 0 and 1
  //   log: true, // log progress periodically 
  //   logPeriod: 100, // related to loggin, here 10 is the iterations
	//   learningRate: 0.3 // learning rate 
  // });
  network.train(trainingData, {
    iterations: 20000,
    log: true,
    logPeriod: 100,
    errorThresh: 0.005
  });
  // create a network path for your learning model
  fs.writeFileSync(networkPath, JSON.stringify(network.toJSON(), null, 2));
}
const output = network.run('survey');
console.log(`Category: ${output}`);





// The network will stop training whenever one of 
// the two criteria is met: the training error has gone 
// below the threshold (default 0.005), or the max number of iterations (default 20000) has been reached.

//The learning rate is a parameter that influences 
//how quickly the network trains. It's a number from 0 to 1. 
// If the learning rate is close to 0, it will take longer to train. 
// If the learning rate is closer to 1, it will train faster, 
// but training results may be constrained to a local minimum and perform badly on new data.


///The training error should decrease every time. 
//The updates will be printed to console. If you set log to a function, 
// this function will be called with the updates instead of printing to the console.