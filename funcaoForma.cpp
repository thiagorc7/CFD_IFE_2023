// Programa para geracao das funcoes de forma e suas matrizes gradientes para Triangulos
// Numeracao original: numeracao em linha dos triangulos de pascal
// Renumeracao para Gmsh: antihoraria, com hierarquia
//tipoNumeracao 0: numeracao linha alta ordem; 1: numeracao antihoraria hierarquica ate 3 ordem; 

#include "funcaoForma.h"
#include <iostream>

//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Dense>
//#include <C:/Users/thiag/Desktop/SOFTWARES/VSCode/Test/Eigen/Sparse>

#include <Eigen/Dense>
#include <Eigen/Sparse>

//  Implementacao do programa funcaoForma incluindo a funcao FuncaoFormaTriangulo
// Para tetraedrico e necessario incluir dphi_dksi3 ou usar apenas dphi_dksi

    void FuncaoFormaTriangulo(Eigen::MatrixXd& coef, Eigen::MatrixXd& dcoef_dksi1,Eigen::MatrixXd& dcoef_dksi2, int grau, Eigen::VectorXd& ksi,
                          Eigen::VectorXd& phi,
                           Eigen::MatrixXd& dphi_dksi) {

    // Inicializacao
     int ndir=2; // funcao fixada para elementos triangulares
      int dim = ((grau + 1) * (grau + 2)) / 2; //vale apenas para triangulos de grau qualquer; dim = nnoselemSuperficie

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Eigen::VectorXd dphi_dksi1 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd dphi_dksi2 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd dphi_dksi3 = Eigen::VectorXd::Zero(dim);
    dphi_dksi = Eigen::MatrixXd::Zero(dim,ndir);
    phi = Eigen::VectorXd::Zero(dim);

    double ksi1=ksi(0); double ksi2=ksi(1);

    // Funções forma
    for (int linha = 0; linha < dim; ++linha) {
        int coluna = 0;
        for (int i = grau + 1; i >= 1; --i) {
            for (int j = 1; j <= i; ++j) {
                coluna++;
                phi(linha) += coef(coluna - 1, linha) * pow(ksi1, (i - j)) * pow(ksi2, (j - 1));
            }
        }
    }

    // 1ª derivadas das funções forma em relação a ksi1
    for (int linha = 0; linha < dim; ++linha) {
        int coluna = 0;
        for (int i = grau + 1; i >= 1; --i) {
            for (int j = 1; j <= i; ++j) {
                coluna++;
                if ((i - j - 1) >= 0) {
                    dphi_dksi1(linha) += dcoef_dksi1(coluna - 1, linha) * pow(ksi1, (i - j - 1)) * pow(ksi2, (j - 1));
                }
            }
        }
    }
    //std::cout << "dphi_dksi2:\n" << dphi_dksi2 << std::endl;
    // 1ª derivadas das funções forma em relação a ksi2
    for (int linha = 0; linha < dim; ++linha) {
        int coluna = 0;
        for (int i = grau + 1; i >= 1; --i) {
            for (int j = 1; j <= i; ++j) {
                coluna++;
                if ((j - 2) >= 0) {
                    dphi_dksi2(linha) += dcoef_dksi2(coluna - 1, linha) * pow(ksi1, (i - j)) * pow(ksi2, (j - 2));
                    //std::cout << "dphi_dksi2 linha:\n" << dphi_dksi2(linha) << std::endl;
                }
            }
        }
    }
    // adc 1ª derivadas das funções forma em relação a ksi3


    //std::cout << "dphi_dksi2:\n" << dphi_dksi2 << std::endl;
    //std::cout << "phi: \n" << phi << std::endl;
    
    //GENERALIZAR PARA QUALQUER 1D OU 2D OU 3D
    //dphi_dksi - Gradiente da funcao Phi 2D
    for (int linha=0; linha <dim; ++linha){
    
            dphi_dksi(linha,0)+=dphi_dksi1(linha);
            dphi_dksi(linha,1)+=dphi_dksi2(linha);

    }

    int tipoNumeracao=1; //tipoNumeracao 0: numeracao linha alta ordem; 1: numeracao antihoraria hierarquica ate 3 ordem; 
    Eigen::VectorXd phiLinha = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd dphiLinha = Eigen::MatrixXd::Zero(dim,ndir);
    phiLinha = phi;
    dphiLinha= dphi_dksi;
    if (tipoNumeracao==1){

    // Renumeracao elemento 3D
    if (ndir==3){
    if(grau==3){

    } else if (grau==2){

    } else if (grau==1){

    }

    // Renumeracao elemento 2D
    } else if (ndir==2){
    if(grau==3){
        phi(0)=phiLinha(0); dphi_dksi(0,0)=dphiLinha(0,0);  dphi_dksi(0,1)=dphiLinha(0,1);
        phi(1)=phiLinha(3); dphi_dksi(1,0)=dphiLinha(3,0);  dphi_dksi(1,1)=dphiLinha(3,1);
        phi(2)=phiLinha(9); dphi_dksi(2,0)=dphiLinha(9,0);  dphi_dksi(2,1)=dphiLinha(9,1);
        phi(3)=phiLinha(1); dphi_dksi(3,0)=dphiLinha(1,0);  dphi_dksi(3,1)=dphiLinha(1,1);
        phi(4)=phiLinha(2); dphi_dksi(4,0)=dphiLinha(2,0);  dphi_dksi(4,1)=dphiLinha(2,1);

        phi(5)=phiLinha(6); dphi_dksi(5,0)=dphiLinha(6,0);  dphi_dksi(5,1)=dphiLinha(6,1);
        phi(6)=phiLinha(8); dphi_dksi(6,0)=dphiLinha(8,0);  dphi_dksi(6,1)=dphiLinha(8,1);
        phi(7)=phiLinha(7); dphi_dksi(7,0)=dphiLinha(7,0);  dphi_dksi(7,1)=dphiLinha(7,1);
        phi(8)=phiLinha(4); dphi_dksi(8,0)=dphiLinha(4,0);  dphi_dksi(8,1)=dphiLinha(4,1);
        phi(9)=phiLinha(5); dphi_dksi(9,0)=dphiLinha(5,0);  dphi_dksi(9,1)=dphiLinha(5,1);


    } else if (grau==2){
        phi(0)=phiLinha(0); dphi_dksi(0,0)=dphiLinha(0,0);  dphi_dksi(0,1)=dphiLinha(0,1);
        phi(1)=phiLinha(2); dphi_dksi(1,0)=dphiLinha(2,0);  dphi_dksi(1,1)=dphiLinha(2,1);
        phi(2)=phiLinha(5); dphi_dksi(2,0)=dphiLinha(5,0);  dphi_dksi(2,1)=dphiLinha(5,1);

        phi(3)=phiLinha(1); dphi_dksi(3,0)=dphiLinha(1,0);  dphi_dksi(3,1)=dphiLinha(1,1);
        phi(4)=phiLinha(4); dphi_dksi(4,0)=dphiLinha(4,0);  dphi_dksi(4,1)=dphiLinha(4,1);
        phi(5)=phiLinha(3); dphi_dksi(5,0)=dphiLinha(3,0);  dphi_dksi(5,1)=dphiLinha(3,1);

    } else if (grau==1){
        //basta manter, pois phi=phiLinha; dphi_dksi=dphiLinha;

    }

    }



    }

}

