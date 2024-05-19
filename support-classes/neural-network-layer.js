import Matrix from './matrix.js';
import functions from './functions.js';

export default class NeuralNetworkLayer {
	constructor(NeuralNetwork, inputNodes, outputNodes) {
		this.NeuralNetwork = NeuralNetwork;

		this.inputNodes = inputNodes;
		this.outputNodes = outputNodes;

		this.weights = new Matrix(outputNodes, inputNodes);
		this.biases = new Matrix(outputNodes, 1);

		this.weights.randomize();
		this.biases.randomize();
	}

	predict(inputMatrix) {
		let outputMatrix = Matrix.multiply(this.weights, inputMatrix);

		outputMatrix.add(this.biases);
		outputMatrix.map(this.NeuralNetwork.activation.activation);

		return outputMatrix;
	}

	calculateAndApplyErrors(outputMatrix, previousOutputMatrix, currentErrors) {
		// Calculate gradients
		let gradients = Matrix.map(
			outputMatrix,
			this.NeuralNetwork.activation.activationDerivative
		);
		gradients.multiply(currentErrors);
		gradients.multiply(this.NeuralNetwork.learningRate);

		// Calculate deltas
		let previousOutputMatrixTransposed =
			Matrix.transpose(previousOutputMatrix);
		let deltas = Matrix.multiply(gradients, previousOutputMatrixTransposed);

		// Add deltas and gradients
		this.weights.add(deltas);
		this.biases.add(gradients);

		// Calculate new errors
		let weightsTransposed = Matrix.transpose(this.weights);
		let newErrors = Matrix.multiply(weightsTransposed, currentErrors);

		return newErrors;
	}

	serialize() {
		return {
			inputNodes: this.inputNodes,
			outputNodes: this.outputNodes,
			weights: this.weights.serialize(),
			biases: this.biases.serialize(),
		};
	}

	static deserialize(network, serializedData) {
		const { inputNodes, outputNodes, weights, biases } = serializedData;
		const layer = new NeuralNetworkLayer(network, inputNodes, outputNodes);

		layer.weights = Matrix.deserialize(JSON.parse(weights));
		layer.biases = Matrix.deserialize(JSON.parse(biases));

		return layer;
	}
}
