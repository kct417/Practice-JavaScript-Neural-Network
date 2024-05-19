export default class Matrix {
	constructor(rows, columns) {
		this.rows = rows;
		this.columns = columns;
		this.data = [];

		for (let i = 0; i < this.rows; i++) {
			this.data[i] = [];
			for (let j = 0; j < this.columns; j++) {
				this.data[i][j] = 0;
			}
		}
	}

	copy() {
		let matrix = new Matrix(this.rows, this.cols);
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				matrix.data[i][j] = this.data[i][j];
			}
		}
		return matrix;
	}

	randomize(range = 2, shift = 1) {
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.columns; j++) {
				this.data[i][j] = Math.random() * range - shift;
			}
		}
	}

	// Create a matrix from an array
	static fromArray(array) {
		let matrix = new Matrix(array.length, 1);
		for (let i = 0; i < array.length; i++) {
			matrix.data[i][0] = array[i];
		}
		return matrix;
	}

	// Create an array from a matrix
	toArray() {
		let array = [];
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.columns; j++) {
				array.push(this.data[i][j]);
			}
		}
		return array;
	}

	static add(matrix1, matrix2) {
		// Matrix addition
		if (
			matrix1.rows != matrix2.rows &&
			matrix1.columns != matrix2.columns
		) {
			console.log('Size of A must match size of B');
			return;
		}
		let result = new Matrix(matrix1.rows, matrix1.columns);
		for (let i = 0; i < result.rows; i++) {
			for (let j = 0; j < result.columns; j++) {
				result.data[i][j] = matrix1.data[i][j] + matrix2.data[i][j];
			}
		}
		return result;
	}

	add(n) {
		if (n instanceof Matrix) {
			// Matrix addition
			if (this.rows != n.rows && this.columns != n.columns) {
				console.log('Size of A must match size of B');
				return;
			}
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.columns; j++) {
					this.data[i][j] += n.data[i][j];
				}
			}
		} else {
			// Scalar addition
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.columns; j++) {
					this.data[i][j] += n;
				}
			}
		}
	}

	static subtract(matrix1, matrix2) {
		// Matrix subtraction
		if (
			matrix1.rows != matrix2.rows &&
			matrix1.columns != matrix2.columns
		) {
			console.log('Size of A must match size of B');
			return;
		}
		let result = new Matrix(matrix1.rows, matrix1.columns);
		for (let i = 0; i < result.rows; i++) {
			for (let j = 0; j < result.columns; j++) {
				result.data[i][j] = matrix1.data[i][j] - matrix2.data[i][j];
			}
		}
		return result;
	}

	subtract(n) {
		if (n instanceof Matrix) {
			// Matrix subtraction
			if (this.rows != n.rows && this.columns != n.columns) {
				console.log('Size of A must match size of B');
				return;
			}
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.columns; j++) {
					this.data[i][j] -= n.data[i][j];
				}
			}
		} else {
			// Scalar subtraction
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.columns; j++) {
					this.data[i][j] -= n;
				}
			}
		}
	}

	static multiply(matrix1, matrix2) {
		// Matrix product
		if (matrix1.columns != matrix2.rows) {
			console.log('Columns of A must match rows of B');
			return;
		}
		let result = new Matrix(matrix1.rows, matrix2.columns);
		for (let i = 0; i < result.rows; i++) {
			for (let j = 0; j < result.columns; j++) {
				let sum = 0;
				for (let k = 0; k < matrix1.columns; k++) {
					sum += matrix1.data[i][k] * matrix2.data[k][j];
				}
				result.data[i][j] = sum;
			}
		}
		return result;
	}

	multiply(n) {
		if (n instanceof Matrix) {
			// Hadamard product
			if (this.rows != n.rows && this.columns != n.columns) {
				console.log('Size of A must match size of B');
				return;
			}
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.columns; j++) {
					this.data[i][j] *= n.data[i][j];
				}
			}
		} else {
			// Scalar product
			for (let i = 0; i < this.rows; i++) {
				for (let j = 0; j < this.columns; j++) {
					this.data[i][j] *= n;
				}
			}
		}
	}

	static map(matrix, func) {
		let result = new Matrix(matrix.rows, matrix.columns);
		for (let i = 0; i < result.rows; i++) {
			for (let j = 0; j < result.columns; j++) {
				let value = matrix.data[i][j];
				result.data[i][j] = func(value);
			}
		}
		return result;
	}

	map(map_function) {
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.columns; j++) {
				let value = this.data[i][j];
				this.data[i][j] = map_function(value);
			}
		}
	}

	static transpose(matrix) {
		let result = new Matrix(matrix.columns, matrix.rows);

		for (let i = 0; i < matrix.rows; i++) {
			for (let j = 0; j < matrix.columns; j++) {
				result.data[j][i] = matrix.data[i][j];
			}
		}
		return result;
	}

	transpose() {
		let result = new Matrix(this.columns, this.rows);
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.columns; j++) {
				result.data[j][i] = this.data[i][j];
			}
		}
		this.rows = result.rows;
		this.columns = result.columns;
		this.data = result.data;
	}

	print() {
		console.table(this.data);
	}

	serialize() {
		return JSON.stringify(this);
	}

	static deserialize(serializedData) {
		const { rows, columns, data } = serializedData;

		const matrix = new Matrix(rows, columns);
		matrix.data = data;

		return matrix;
	}
}
