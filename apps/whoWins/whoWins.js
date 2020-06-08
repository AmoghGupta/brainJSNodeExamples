const brain = require('brain.js');

const network = new brain.NeuralNetwork();

// if o/p is 1,  2nd player wins
// if o/p is 0, 1st player wins

network.train([
  { input: [1, 2], output: [1] }, // Team 2 wins
  { input: [1, 3], output: [1] }, // Team 3 wins
  { input: [2, 3], output: [0] }, // Team 2 wins
  { input: [2, 4], output: [1] }, // Team 4 wins
  { input: [1, 2], output: [0] }, // Team 1 wins
  { input: [1, 3], output: [0] }, // Team 1 wins
  { input: [3, 4], output: [0] } // Team 3 wins
]);

const output = network.run([1, 4]);

console.log(`Prob: ${output}`);
