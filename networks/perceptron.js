import Matrix from '../support-classes/matrix.js';
import functions from '../support-classes/functions.js';

export default class Perceptron {
	constructor(inputNodes) {
		this.inputNodes = inputNodes;

		this.weights = new Matrix(1, this.inputNodes);
		this.bias = new Matrix(1, 1);

		this.weights.randomize();
		this.bias.randomize();

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
		let inputMatrix = Matrix.fromArray(inputArray);

		let outputMatrix = Matrix.multiply(this.weights, inputMatrix);
		outputMatrix.add(this.bias);
		outputMatrix.map(this.activation.activation);

		return outputMatrix.toArray();
	}

	train(inputArray, targetArray) {
		let inputMatrix = Matrix.fromArray(inputArray);
		let targetMatrix = Matrix.fromArray(targetArray);

		let outputArray = this.predict(inputArray);
		let outputMatrix = Matrix.fromArray(outputArray);
		outputMatrix.add(this.bias);
		let outputResults = Matrix.map(
			outputMatrix,
			this.activation.activation
		);

		let errors = Matrix.subtract(targetMatrix, outputResults);

		let gradients = Matrix.map(
			outputResults,
			this.activation.activationDerivative
		);
		gradients.multiply(errors);
		gradients.multiply(this.learningRate);

		let inputTransposed = Matrix.transpose(inputMatrix);
		let deltas = Matrix.multiply(gradients, inputTransposed);

		this.weights.add(deltas);
		this.bias.add(gradients);
	}
}
