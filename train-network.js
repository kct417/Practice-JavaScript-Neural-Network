import fs from 'fs';
import MultiLayerNetwork from './networks/multi-layer-network.js';
// import functions from './support-classes/functions.js';

const filePath = 'serializedData.json';

function main() {
	const args = process.argv.slice(2);
	if (args.length < 1 || args.length > 2) {
		console.log(
			'Usage: node train-network.js <trainingdataFile> [<iterations>]'
		);
		process.exit(1);
	}

	// Extract command line arguments
	const trainingDataFile = args[0];
	const iterations = args.length === 2 ? parseInt(args[1]) : 100000;

	// Extract neural network parameters
	const [, networkParams] = trainingDataFile.match(/(\d+-\d+(-\d+)+)\.json/);
	const networkParamsArray = networkParams.split('-').map(Number);
	const inputNodes = networkParamsArray.shift();
	const outputNodes = networkParamsArray.pop();
	const hiddenNodes = networkParamsArray;

	const filePath = `serialized-data\\${networkParams}.json`;

	// Read training data from file
	let trainingData = readFile(trainingDataFile);

	let network;
	if (fs.existsSync(filePath)) {
		network = MultiLayerNetwork.deserializeFromFile(filePath);
	} else {
		network = new MultiLayerNetwork(inputNodes, hiddenNodes, outputNodes);
	}
	// network.setLearningRate(0.01);
	// network.setActivationFunction(functions.sigmoid);

	// Train neural network
	train(network, trainingData, iterations);

	// Test neural network
	test(network, trainingData);

	network.serializeToFile(filePath);
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
