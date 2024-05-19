class ActivationFunction {
	constructor(activation, activationDerivative) {
		this.activation = activation;
		this.activationDerivative = activationDerivative;
	}

	serialize() {
		return {
			activation: this.activation.toString(),
			activationDerivative: this.activationDerivative.toString(),
		};
	}

	static deserialize(serializedData) {
		const { activation, activationDerivative } = serializedData;

		const activationFunction = new Function(`return ${activation}`)();
		const activationFunctionDerivative = new Function(
			`return ${activationDerivative}`
		)();

		return new ActivationFunction(
			activationFunction,
			activationFunctionDerivative
		);
	}
}

const sigmoid = new ActivationFunction(
	(x) => 1 / (1 + Math.exp(-x)),
	(y) => y * (1 - y)
);

const tanh = new ActivationFunction(
	(x) => Math.tanh(x),
	(y) => 1 - y * y
);

const relu = new ActivationFunction(
	(x) => Math.max(0, x),
	(y) => (y > 0 ? 1 : 0)
);

const leakyRelu = new ActivationFunction(
	(x) => (x > 0 ? x : 0.01 * x),
	(y) => (y > 0 ? 1 : 0.01)
);

const elu = new ActivationFunction(
	(x) => (x >= 0 ? x : Math.exp(x) - 1),
	(y) => (y >= 0 ? 1 : y + 1)
);

export default {
	ActivationFunction,
	sigmoid,
	tanh,
	relu,
	leakyRelu,
	elu,
};
