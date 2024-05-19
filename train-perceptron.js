import fs from 'fs';
import Perceptron from './networks/perceptron.js';
// import functions from './support-classes/functions.js';

function main() {
	const args = process.argv.slice(2);
	if (args.length < 1 || args.length > 2) {
		console.log(
			'Usage: node train-perceptron.js <trainingdataFile> [<iterations>]'
		);
		process.exit(1);
	}

	// Extract command line arguments
	const trainingDataFile = args[0];
	const iterations = args.length === 2 ? parseInt(args[1]) : 100000;

	// Extract neural network parameters
	const [, networkParams] = trainingDataFile.match(/(\d)\.json/);
	const inputNodes = networkParams.split().map(Number);

	// Read training data from file
	let trainingData = readFile(trainingDataFile);

	let network = new Perceptron(inputNodes);
	// network.setLearningRate(0.01);
	// network.setActivationFunction(functions.sigmoid);

	// Train neural network
	train(network, trainingData, iterations);

	// Test neural network
	test(network, trainingData);
}

function readFile(trainingDataFile) {
	let trainingData;
	try {
		const data = fs.readFileSync(trainingDataFile, 'utf8');
		trainingData = JSON.parse(data);
	} catch (error) {
		console.error(
			`Error reading training data file \'${trainingDataFile}\':`,
			error
		);
		process.exit(1);
	}

	return trainingData;
}

function train(network, trainingData, iterations) {
	for (let i = 0; i < iterations; i++) {
		let randomIndex = Math.floor(Math.random() * trainingData.length);
		let data = trainingData[randomIndex];
		network.train(data.inputs, data.targets);
	}
}

function test(network, trainingData) {
	for (const data of trainingData) {
		console.log(
			data.inputs,
			':',
			data.targets,
			':',
			network.predict(data.inputs)
		);
	}
}

main();
