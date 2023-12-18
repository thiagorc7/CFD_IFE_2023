//arquivo de cabecalho para a funcao: funcaoForma.h funcaoFormaLagrange.h
#ifndef FUNCAO_FORMA_LAGRANGE_H
#define FUNCAO_FORMA_LAGRANGE_H

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Formato de chamada das funcoes do programa funcaoForma.cpp
void FuncaoFormaLinha(int tipoFuncaoForma,int ndir, int grau, Eigen::VectorXd& ksi,
                          Eigen::VectorXd& phi,
                           Eigen::MatrixXd& dphi_dksi);

#endif