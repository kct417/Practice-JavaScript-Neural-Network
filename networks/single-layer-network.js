import Matrix from '../support-classes/matrix.js';
import functions from '../support-classes/functions.js';

export default class SingleLayerNetwork {
	constructor(inputNodes, hiddenNodes, outputNodes) {
		this.inputNodes = inputNodes;
		this.hiddenNodes = hiddenNodes;
		this.outputNodes = outputNodes;

		this.inputHiddenWeights = new Matrix(this.hiddenNodes, this.inputNodes);
		this.hiddenOutputWeights = new Matrix(
			this.outputNodes,
			this.hiddenNodes
		);

		this.hiddenBiases = new Matrix(this.hiddenNodes, 1);
		this.outputBiases = new Matrix(this.outputNodes, 1);

		this.inputHiddenWeights.randomize();
		this.hiddenOutputWeights.randomize();
		this.hiddenBiases.randomize();
		this.outputBiases.randomize();

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

		// Calculate hiddens
		let hiddenMatrix = Matrix.multiply(
			this.inputHiddenWeights,
			inputMatrix
		);
		hiddenMatrix.add(this.hiddenBiases);
		hiddenMatrix.map(this.activation.activation);

		// Calculate outputs
		let outputMatrix = Matrix.multiply(
			this.hiddenOutputWeights,
			hiddenMatrix
		);
		outputMatrix.add(this.outputBiases);
		outputMatrix.map(this.activation.activation);

		return outputMatrix.toArray();
	}

	train(inputArray, targetArray) {
		// Feedforward

		let inputMatrix = Matrix.fromArray(inputArray);
		let targetMatrix = Matrix.fromArray(targetArray);

		// Calculate hiddens
		let hiddenMatrix = Matrix.multiply(
			this.inputHiddenWeights,
			inputMatrix
		);
		hiddenMatrix.add(this.hiddenBiases);
		let hiddenResults = Matrix.map(
			hiddenMatrix,
			this.activation.activation
		);

		// Calculate outputs
		let outputMatrix = Matrix.multiply(
			this.hiddenOutputWeights,
			hiddenResults
		);
		outputMatrix.add(this.outputBiases);
		let outputResults = Matrix.map(
			outputMatrix,
			this.activation.activation
		);

		// Backpropagation

		// Calculate output errors
		let outputErrors = Matrix.subtract(targetMatrix, outputResults);

		// Calculate output gradients
		let outputGradients = Matrix.map(
			outputResults,
			this.activation.activationDerivative
		);
		outputGradients.multiply(outputErrors);
		outputGradients.multiply(this.learningRate);

		// Calculate output deltas
		let hiddenTransposed = Matrix.transpose(hiddenResults);
		let hiddenOutputWeightDeltas = Matrix.multiply(
			outputGradients,
			hiddenTransposed
		);

		// Add output deltas to hidden output weights
		this.hiddenOutputWeights.add(hiddenOutputWeightDeltas);
		// Add output deltas to output biases which is just output gradients
		this.outputBiases.add(outputGradients);

		// Caclulate hidden errors
		let hiddenOutputWeights_transposed = Matrix.transpose(
			this.hiddenOutputWeights
		);
		let hiddenErrors = Matrix.multiply(
			hiddenOutputWeights_transposed,
			outputErrors
		);

		// Calculate hidden gradients
		let hiddenGradients = Matrix.map(
			hiddenResults,
			this.activation.activationDerivative
		);
		hiddenGradients.multiply(hiddenErrors);
		hiddenGradients.multiply(this.learningRate);

		// Calculate hidden deltas
		let inputTransposed = Matrix.transpose(inputMatrix);
		let inputHiddenWeightDeltas = Matrix.multiply(
			hiddenGradients,
			inputTransposed
		);

		// Add hidden deltas to input hidden weights
		this.inputHiddenWeights.add(inputHiddenWeightDeltas);
		// Add hidden deltas to hidden biases which is just hidden gradients
		this.hiddenBiases.add(hiddenGradients);
	}
}
