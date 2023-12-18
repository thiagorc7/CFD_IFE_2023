// Programa para geracao dos coeficientes para malha de Triangulos
// Numeracao original: numeracao em linha dos triangulos de pascal
// Renumeracao para Gmsh: antihoraria, com hierarquia
//tipoNumeracao 0: numeracao linha alta ordem; 1: numeracao antihoraria hierarquica ate 3 ordem; 

#include "funcaoCoeficientesTriangular.h"
#include <iostream>

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//  Implementacao do programa funcaoForma incluindo a funcao FuncaoFormaTriangulo
// Para tetraedrico e necessario incluir dphi_dksi3 ou usar apenas dphi_dksi

    void funcaoCoeficientesTriangular(Eigen::MatrixXd& coef, Eigen::MatrixXd& dcoef_dksi1,Eigen::MatrixXd& dcoef_dksi2, int grau) {

    // Inicializacao
      int dim = ((grau + 1) * (grau + 2)) / 2; //vale apenas para triangulos de grau qualquer; dim = nnoselemSuperficie
    //tipoFuncaoForma=1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;
    Eigen::VectorXd adm1(dim);
    Eigen::VectorXd adm2(dim);
    Eigen::MatrixXd matriz(dim, dim);

    coef = Eigen::MatrixXd::Zero(dim,dim); // retorna
    dcoef_dksi1 = Eigen::MatrixXd::Zero(dim,dim); // retorna
    dcoef_dksi2 = Eigen::MatrixXd::Zero(dim,dim); // retorna
    
    // Coordenadas adimensionais
    int no = 0;
    for (int j = 1; j <= (grau + 1); ++j) {
        for (int i = 1; i <= (grau + 1) - (j - 1); ++i) {
            no++;
            adm1(no - 1) = 0.0 + (i - 1) * (1.0 / (1.0 * grau));
            adm2(no - 1) = 0.0 + (j - 1) * (1.0 / (1.0 * grau));
        }
    }
    
    // Matriz dos coeficientes
    for (int linha = 0; linha < dim; ++linha) {
        int coluna = 0;
        for (int i = grau + 1; i >= 1; --i) {
            for (int j = 1; j <= i; ++j) {
                coluna++;
                matriz(linha, coluna - 1) = pow(adm1(linha), (i - j)) * pow(adm2(linha), (j - 1));
            }
        }
    }
    //std::cout << "Matriz M:\n" << matriz << std::endl;

    // Obtencao da matriz dos coeficientes a partir da inversao de M
    coef = matriz.inverse();
    //std::cout << "Matriz dos Coeficientes do Elemento:\n" << coef << std::endl;

    // 1ª derivada dos coeficientes em relação a ksi1 analitico
    for (int linha = 0; linha < dim; ++linha) {
        int coluna = 0;
        for (int i = grau + 1; i >= 1; --i) {
            for (int j = 1; j <= i; ++j) {
                coluna++;
                dcoef_dksi1(coluna - 1, linha) = coef(coluna - 1, linha) * (1.0 * i - 1.0 * j);
            }
        }
    }

    // 1ª derivada dos coeficientes em relação a ksi2
    for (int linha = 0; linha < dim; ++linha) {
        int coluna = 0;
        for (int i = grau + 1; i >= 1; --i) {
            for (int j = 1; j <= i; ++j) {
                coluna++;
                dcoef_dksi2(coluna - 1, linha) = coef(coluna - 1, linha) * (1.0 * j - 1.0);
            }
        }
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 

} // final void da funcao

