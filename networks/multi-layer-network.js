import fs from 'fs';
import Matrix from '../support-classes/matrix.js';
import functions from '../support-classes/functions.js';
import NeuralNetworkLayer from '../support-classes/neural-network-layer.js';

export default class MultiLayerNetwork {
	constructor(inputNodes, hiddenNodes, outputNodes) {
		this.inputNodes = inputNodes;
		this.outputNodes = outputNodes;

		if (Number.isInteger(hiddenNodes)) {
			this.hiddenNodes = [hiddenNodes];
		} else {
			this.hiddenNodes = hiddenNodes;
		}

		this.layers = [];

		// Input hidden layer
		this.layers.push(
			new NeuralNetworkLayer(this, this.inputNodes, this.hiddenNodes[0])
		);

		// Hidden hidden layers
		for (let i = 1; i < this.hiddenNodes.length; i++) {
			this.layers.push(
				new NeuralNetworkLayer(
					this,
					this.hiddenNodes[i - 1],
					this.hiddenNodes[i]
				)
			);
		}

		// Hidden output layer
		this.layers.push(
			new NeuralNetworkLayer(
				this,
				this.hiddenNodes[this.hiddenNodes.length - 1],
				this.outputNodes
			)
		);

		this.setLearningRate();
		this.setActivationFunction();
	}

	setLearningRate(learningRate = 0.1) {
		this.learningRate = learningRate;
	}

	setActivationFunction(activation = functions.sigmoid) {
		this.activation = activation;
	}

	predict(inputArray) {
		// Feedforward

		let inputMatrix = Matrix.fromArray(inputArray);

		let outputMatrix = inputMatrix;
		for (let i = 0; i < this.layers.length; i++) {
			outputMatrix = this.layers[i].predict(outputMatrix);
		}

		return outputMatrix.toArray();
	}

	train(inputArray, targetArray) {
		// Feedforward

		let inputMatrix = Matrix.fromArray(inputArray);
		let targetMatrix = Matrix.fromArray(targetArray);

		let outputMatrices = [];
		let outputMatrix = inputMatrix;

		// Calculate hiddens
		for (let i = 0; i < this.layers.length; i++) {
			outputMatrix = this.layers[i].predict(outputMatrix);
			outputMatrices.push(outputMatrix);
		}

		// Calculate outputs
		outputMatrix = outputMatrices[outputMatrices.length - 1];

		// Backpropagation

		let currentErrors = Matrix.subtract(targetMatrix, outputMatrix);

		// Calculate and apply errors
		for (let i = this.layers.length - 1; i > 0; i--) {
			currentErrors = this.layers[i].calculateAndApplyErrors(
				outputMatrices[i],
				outputMatrices[i - 1],
				currentErrors
			);
		}

		// Apply errors for first hidden layers
		this.layers[0].calculateAndApplyErrors(
			outputMatrices[0],
			inputMatrix,
			currentErrors
		);
	}

	serializeToFile(filename) {
		const serializedData = JSON.stringify(this.serialize());
		fs.writeFileSync(filename, serializedData);
		console.log(`Data saved to \'${filename}\'`);
	}

	static deserializeFromFile(filename) {
		const serializedData = fs.readFileSync(filename, 'utf8');
		console.log(`Data retreived from \'${filename}\'`);
		return MultiLayerNetwork.deserialize(JSON.parse(serializedData));
	}

	serialize() {
		console.log('Serializing data');
		const serializedLayers = this.layers.map((layer) => layer.serialize());
		return {
			inputNodes: this.inputNodes,
			hiddenNodes: this.hiddenNodes,
			outputNodes: this.outputNodes,
			layers: serializedLayers,
			learningRate: this.learningRate,
			activation: this.activation.serialize(),
		};
	}

	static deserialize(serializedData) {
		console.log('Deserializing data');
		const {
			inputNodes,
			hiddenNodes,
			outputNodes,
			layers,
			learningRate,
			activation,
		} = serializedData;
		const network = new MultiLayerNetwork(
			inputNodes,
			hiddenNodes,
			outputNodes
		);

		network.layers = layers.map((layer) =>
			NeuralNetworkLayer.deserialize(network, layer)
		);

		network.setLearningRate(learningRate);
		network.setActivationFunction(
			functions.ActivationFunction.deserialize(activation)
		);

		return network;
	}
}
