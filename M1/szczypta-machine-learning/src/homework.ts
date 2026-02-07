import { addMatrices, multiplyMatrices, transpose, assertMatricesDimensionMatch, assertMatricesCompatible } from "./matrix-operations";
import { fromJSONFile, jsonFilePath, randomizeMatrix, randomizeVector } from "./utils";
import { vectorSum, dotProduct } from "./vector-operations";
import { Matrix, Vector } from "./types";
import { displayVector, displayMatrix } from "./display";

// HINT: (w zaleności od wybranego kierunku implementacji) może być mnożenie macierzy przez wektory - tę operację będzie trzeba zaimplementować 😉
// ale nie jest to konieczne 😎

// HINT: w mnożeniu macierzy kolejność ma znaczenie - bo w zależności od kolejności albo wymiary obydwu składników pasują do siebie albo nie.

// HINT: wstań od komputera i przemyśl problem. Serio. Zastanów się, ile linijek wystarczy aby podać rozwiązanie :)
// (traktując "linijkę" jako pojedynczą operację na tensorach) 😎

// PROŚBA: jeśli znasz rozwiązanie, to nie spamuj discorda - a przynajmniej nie od razu. Pozwól innym pomóżdżyć 😎

// const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-1.json'));
// const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-2.json'));
// const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-3.json'));
const { WK_Matrix, WQ_Matrix, X_Input_Matrix } = fromJSONFile(jsonFilePath('case-4.json'));

console.log('WK_Matrix');
console.log(displayMatrix(WK_Matrix, -1));
console.log('WQ_Matrix');
console.log(displayMatrix(WQ_Matrix, -1));
console.log('X_Input_Matrix');
console.log(displayMatrix(X_Input_Matrix, -1));

const x1_vector = X_Input_Matrix[0];
console.log('x1_vector');
console.log(displayVector(x1_vector, -1));

// przypomnienie zadania: naley policzyć "attention matrix S"
// const x1_k_vector = WK_Matrix.map((row) => {
//     return dotProduct(row, x1_vector)
// });
// console.log('x1_k_vector')
// console.log(displayVector(x1_k_vector, -1));

// const x1_q_vector = WQ_Matrix.map((row) => {
//     return dotProduct(row, x1_vector)
// });
// console.log('x1_q_vector')
// console.log(displayVector(x1_q_vector, -1));

// const x1_attention_score = dotProduct(x1_q_vector, x1_k_vector);
// console.log('x1_attention_score')
// console.log(displayVector([x1_attention_score], -1))

// const attention_score_matrix_for_x1_vector = multiplyMatrices(WQ_Matrix, transpose(WK_Matrix))
// console.log('attention_score_matrix');
// console.log(displayMatrix(attention_score_matrix_for_x1_vector));
const Q = multiplyMatrices(X_Input_Matrix, WQ_Matrix);
console.log('Q')
console.log(displayMatrix(Q, -1));

const K = multiplyMatrices(X_Input_Matrix, WK_Matrix);
console.log('K')
console.log(displayMatrix(K, -1));

const attention_score_matrix = multiplyMatrices(Q, transpose(K));
console.log('attention_score_matrix');
console.log(displayMatrix(attention_score_matrix, -1));