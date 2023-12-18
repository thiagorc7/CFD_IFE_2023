//------------------------------------------------------------------------------
// 
//                   Jeferson W D Fernandes and Rodolfo A K Sanches
//                             University of Sao Paulo
//                           (C) 2017 All Rights Reserved
//
// <LicenseText>
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------FLUID-------------------------------------
//------------------------------------------------------------------------------

#ifndef FLUID_H
#define FLUID_H

#include "Element.hpp"
#include "Boundary.hpp"
#include "fluidDomain.h"

// PETSc libraries
#include <metis.h>
//#include <petscksp.h> 

#include <boost/timer.hpp> 
#include <boost/thread.hpp>


// Inclusao de bibliotecas auxiliares padrao C++
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


// Inclusao de programas auxiliares ou cabecalhos ou headers
#include "propGeo.h"
#include "funcaoForma.h"
#include "funcaoFormaLagrange.h"
#include "funcaoCoeficientesTriangular.h"

// Inclusoes especificas do PETSc
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>

// Inclusoes do MPI
#include <mpi.h>


// Eigen bibliotecas
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <algorithm>
#include <iomanip> // Para std::setprecision


// name space
using namespace Eigen;
using namespace std;

/// Mounts the incompressible flow problem
template<int DIM>
class Fluid{
public:
    /// Defines the class Element locally
    typedef Element<DIM> Elements;

    /// Defines the class Node locally
    typedef typename Elements::Nodes  Node;

    /// Defines the class Boundary locally
    typedef Boundary<DIM> Boundaries;

    /// Defines the vector of fluid nodes
    std::vector<Node *>       nodes_;

    /// Defines the vector of fluid elements
    std::vector<Elements *>   elements_;
 
    /// Defines the vector of fluid boundaries mesh nodes
    std::vector<Boundaries *> boundary_;

private:
    //FLUID VARIABLES
    std::string inputFile; //Fluid input file
    int numElem;           //Number of elements in fluid mesh for each thread 
    int numTotalElem;           //Total number of elements in fluid mesh 
    int numNodes;          //Number of nodes in velocity/quadratic mesh
    int numBoundaries;     //Number of fluid boundaries
    int numBoundElems;     //Number of elements in fluid boundaries
    double pressInf;       //Undisturbed pressure 
    double rhoInf;         //Density
    double viscInf;        //Viscosity
    double velocityInf[3]; //Undisturbed velocity
    double fieldForces[3]; //Field forces (constant)
    idx_t* part_elem;      //Fluid Domain Decomposition - Elements
    idx_t* part_nodes;     //Fluid Domain Decomposition - Nodes
    int numTimeSteps;      //Number of Time Steps
    int printFreq;         //Printing frequence of output files
    double dTime;          //Time Step
    double integScheme;    //Time Integration Scheme
    int rank;
    int size;
    bool computeDragAndLift;
    int numberOfLines;
    std::vector<int> dragAndLiftBoundary;
    int iTimeStep;
    Geometry* geometry_;
    double pi = M_PI;


    
public:
    int weightFunctionBehavior;
    bool printVelocity;
    bool printPressure;
    bool printVorticity;
    bool printMeshVelocity;
    bool printMeshDisplacement;
    bool printJacobian;
    bool printProcess;

public:
    /// Reads the input file and perform the preprocessing operations
    /// @param Geometry* mesh geometry @param std::string input file 
    /// @param std::string input .msh file @param std::string mirror file
    /// @param bool delete mesh files
    void dataReading(Geometry* geometry, const std::string& inputFile, const std::string& inputMesh, const std::string& mirror, const bool& deleteFiles);

    void readInitialValues(const std::string& inputVel,const std::string& inputPres);


    /// Performs the domain decomposition for parallel processing
    void domainDecompositionMETIS(std::vector<Elements *> &elem_); 

    /// Export the domain decomposition 
    /// @return pair with the elements and nodes domain decompositions
    std::pair<idx_t*,idx_t*> getDomainDecomposition(){
        return std::make_pair(part_elem,part_nodes);};
    
    /// Gets the flag for computing the Drag and Lift coefficients in a 
    /// specific boundary
    /// @return bool flag for computing Drag and Lift coefficients
    bool getComputeDragAndLift() {return computeDragAndLift;};

    /// Gets the number of the boundary for computing the Drag and Lift 
    /// coefficients
    /// @return int boundary number
    int getDragAndLiftBoundary(int index) {return dragAndLiftBoundary[index];};
 
    /// Mounts and solve the transient incompressible flow problem    
    /// @param int maximum number of Newton-Raphson's iterations
    /// @param double tolerance of the Newton-Raphson's process
    int solveTransientProblem(int iterNumber,double tolerance);

    /// Mounts and solve the transient incompressible flow problem for moving
    /// domain problems
    /// @param int maximum number of Newton-Raphson's iterations
    /// @param double tolerance of the Newton-Raphson's process
    int solveTransientProblemMoving(int iterNumber,double tolerance);

    /// Mounts and solve the steady Laplace problem
    /// @param int maximum number of Newton-Raphson's iterations
    /// @param double tolerance of the Newton-Raphson's process
    int solveSteadyLaplaceProblem(int iterNumber, double tolerance);

    /// Print the results for Paraview post-processing
    /// @param int time step
    void printResults(int step);

    /// Compute and print drag and lift coefficients
    void dragAndLiftCoefficients(std::ofstream& dragLift);
};

//------------------------------------------------------------------------------
//--------------------------------IMPLEMENTATION--------------------------------
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//---------------------SUBDIVIDES THE FINITE ELEMENT DOMAIN---------------------
//------------------------------------------------------------------------------
template<>
void Fluid<2>::domainDecompositionMETIS(std::vector<Elements *> &elem_) {
    
    std::string mirror2;
    mirror2 = "domain_decomposition.txt";
    std::ofstream mirrorData(mirror2.c_str());
    
    int size;

    MPI_Comm_size(PETSC_COMM_WORLD, &size);



    idx_t objval;
    idx_t numEl = numElem;
    idx_t numNd = numNodes;
    idx_t dd = 2;
    idx_t ssize = size;
    idx_t three = 3;
    idx_t one = 1;
    idx_t elem_start[numEl+1], elem_connec[(4*dd-2)*numEl];

    MPI_Bcast(&numEl,1,MPI_INT,0,PETSC_COMM_WORLD);
    MPI_Bcast(&numNd,1,MPI_INT,0,PETSC_COMM_WORLD);


    part_elem = new idx_t[numEl];
    part_nodes = new idx_t[numNd];


    if (rank == 0){
        for (idx_t i = 0; i < numEl+1; i++){
            elem_start[i]=(4*dd-2)*i;
        };
        for (idx_t jel = 0; jel < numEl; jel++){
            typename Elements::Connectivity connec;
            connec=elem_[jel]->getConnectivity();        
            
            for (idx_t i=0; i<(4*dd-2); i++){
            elem_connec[(4*dd-2)*jel+i] = connec(i);
            };
        };

        //Performs the domain decomposition
        METIS_PartMeshDual(&numEl, &numNd, elem_start, elem_connec,
                                  NULL, NULL, &one, &ssize, NULL, NULL,
                                  &objval, part_elem, part_nodes);

        mirrorData << std::endl 
                   << "FLUID MESH DOMAIN DECOMPOSITION - ELEMENTS" << std::endl;
        for(int i = 0; i < numElem; i++){
            mirrorData << "process = " << part_elem[i]
                       << ", element = " << i << std::endl;
        };

        mirrorData << std::endl 
                   << "FLUID MESH DOMAIN DECOMPOSITION - NODES" << std::endl;
        for(int i = 0; i < numNodes; i++){
            mirrorData << "process = " << part_nodes[i]
                       << ", node = " << i << std::endl;
        };
        

        for (int i = 0; i < size; ++i){
            std::string result;
            std::ostringstream convert;

            convert << i+000;
            result = convert.str();
            std::string s = "mesh"+result+".dat";

            std::fstream mesh(s.c_str(), std::ios_base::out);

            int locElem = std::count(part_elem, part_elem+numElem, i);

            mesh << locElem << std::endl;

            for (int jel = 0; jel < numElem; ++jel){
                if (part_elem[jel] == i){
                    typename Elements::Connectivity connec;
                    connec = elem_[jel]->getConnectivity();
                    mesh << jel << " " << connec(0) << " " << connec(1) << " " << connec(2) << " "
                         << connec(3) << " " << connec(4) << " " << connec(5) << std::endl;
                }
            }

        }
    }

    MPI_Bcast(part_elem,numEl,MPI_INT,0,PETSC_COMM_WORLD);
    MPI_Bcast(part_nodes,numNd,MPI_INT,0,PETSC_COMM_WORLD);

    return;

};

//------------------------------------------------------------------------------
//----------------------------PRINT VELOCITY RESULTS----------------------------
//------------------------------------------------------------------------------
template<>
void Fluid<2>::printResults(int step) {

    std::string s;
    
    if (step % printFreq == 0){

        std::string result;
        std::ostringstream convert;

        convert << step+100000;
        result = convert.str();
        if (rank == 0) s = "saidaVel"+result+".vtu";

        std::fstream output_v(s.c_str(), std::ios_base::out);

        if (rank == 0){
            output_v << "<?xml version=\"1.0\"?>" << std::endl
                     << "<VTKFile type=\"UnstructuredGrid\">" << std::endl
                     << "  <UnstructuredGrid>" << std::endl
                     << "  <Piece NumberOfPoints=\"" << numNodes
                     << "\"  NumberOfCells=\"" << numTotalElem
                     << "\">" << std::endl;

            //WRITE NODAL COORDINATES
            output_v << "    <Points>" << std::endl
                     << "      <DataArray type=\"Float64\" "
                     << "NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;

            for (int i=0; i<numNodes; i++){
                typename Node::VecLocD x;
                x=nodes_[i]->getCoordinates();
                output_v << x(0) << " " << x(1) << " " << 0.0 << std::endl;        
            };
            output_v << "      </DataArray>" << std::endl
                     << "    </Points>" << std::endl;
            
            //WRITE ELEMENT CONNECTIVITY
            output_v << "    <Cells>" << std::endl
                     << "      <DataArray type=\"Int32\" "
                     << "Name=\"connectivity\" format=\"ascii\">" << std::endl;
        }

        int k = 0;
        for (int iElem = 0; iElem < numTotalElem; ++iElem){
            typename Elements::Connectivity connec;

            if (part_elem[iElem] == rank){
                connec = elements_[k]->getConnectivity();
                k++;
            }
            MPI_Bcast(&connec,8,MPI_INT,part_elem[iElem],PETSC_COMM_WORLD);
            if (rank == 0) output_v << connec(0) << " " << connec(1) << " " << connec(2) << " "
                                    << connec(3) << " " << connec(4) << " " << connec(5) << std::endl;

            MPI_Barrier(PETSC_COMM_WORLD);
        }

        if (rank == 0) {
            output_v << "      </DataArray>" << std::endl;
      
            //WRITE OFFSETS IN DATA ARRAY
            output_v << "      <DataArray type=\"Int32\""
                    << " Name=\"offsets\" format=\"ascii\">" << std::endl;
        
            int aux = 0;
            for (int i=0; i<numTotalElem; i++){
                output_v << aux + 6 << std::endl;
                aux += 6;
            };
            output_v << "      </DataArray>" << std::endl;
      
            //WRITE ELEMENT TYPES
            output_v << "      <DataArray type=\"UInt8\" Name=\"types\" "
                     << "format=\"ascii\">" << std::endl;
        
            for (int i=0; i<numTotalElem; i++){
                output_v << 22 << std::endl;
            };

            output_v << "      </DataArray>" << std::endl
                     << "    </Cells>" << std::endl;

            //WRITE NODAL RESULTS
            output_v << "    <PointData>" << std::endl;

            if (printVelocity){
                output_v<< "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                        << "Name=\"Velocity\" format=\"ascii\">" << std::endl;
                for (int i=0; i<numNodes; i++){
                    output_v << nodes_[i] -> getVelocity(0) << " "             
                             << nodes_[i] -> getVelocity(1) << " " << 0. << std::endl;
                }; 
                output_v << "      </DataArray> " << std::endl;
            };

            if (printMeshVelocity){
                output_v<< "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                        << "Name=\"Mesh Velocity\" format=\"ascii\">" << std::endl;
            
                for (int i=0; i<numNodes; i++){
                    output_v << nodes_[i] -> getMeshVelocity(0) << " "    
                             << nodes_[i] -> getMeshVelocity(1) << " " 
                             << 0. << std::endl;
                };
                output_v << "      </DataArray> " << std::endl;
            };

            if (printVorticity){
                output_v <<"      <DataArray type=\"Float64\" NumberOfComponents=\"1\" "
                         << "Name=\"Vorticity\" format=\"ascii\">" << std::endl;
                for (int i=0; i<numNodes; i++){
                    output_v << nodes_[i] -> getVorticity() << std::endl;
                };
                output_v << "      </DataArray> " << std::endl;
            };

            if (printPressure){
                output_v <<"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                         << "Name=\"Pressure\" format=\"ascii\">" << std::endl;
                for (int i=0; i<numNodes; i++){
                    output_v << 0. << " " << 0. << " " 
                             << nodes_[i] -> getPressure() << std::endl;
                };
                output_v << "      </DataArray> " << std::endl;
            };

            if (printMeshDisplacement){
                output_v <<"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                         << "Name=\"Mesh Displacement\" format=\"ascii\">" << std::endl;
                for (int i=0; i<numNodes; i++){       
                    typename Node::VecLocD x, xp;
                    x=nodes_[i]->getCoordinates();
                    xp=nodes_[i]->getInitialCoordinates();
                
                    output_v << x(0)-xp(0) << " " << x(1)-xp(1) << " " 
                            << 0. << std::endl;
                };
                output_v << "      </DataArray> " << std::endl;
            };

                           output_v <<"      <DataArray type=\"Float64\" NumberOfComponents=\"3\" "
                         << "Name=\"Boundary Type\" format=\"ascii\">" << std::endl;
                for (int i=0; i<numNodes; i++){       
                    int x, xp;
                    x=nodes_[i]->getConstrains(0);
                    xp=nodes_[i]->getConstrains(1);
                
                    output_v << x << " " << xp << " " 
                            << 0. << std::endl;
                };
                output_v << "      </DataArray> " << std::endl;



            output_v << "    </PointData>" << std::endl; 

            //WRITE ELEMENT RESULTS
            output_v << "    <CellData>" << std::endl;
        };


        if (printProcess){
            if(rank == 0){
                output_v <<"      <DataArray type=\"Float64\" NumberOfComponents=\"1\" "
                         << "Name=\"Process\" format=\"ascii\">" << std::endl;
            
                for (int i=0; i<numTotalElem; i++){
                    output_v << part_elem[i] << std::endl;
                };
                output_v << "      </DataArray> " << std::endl;
            };
        };

        if (printJacobian){
            if (rank == 0){
                output_v <<"      <DataArray type=\"Float64\" NumberOfComponents=\"1\" "
                         << "Name=\"Jacobian\" format=\"ascii\">" << std::endl;
            };
            

            int k = 0;
            for (int iElem = 0; iElem < numTotalElem; ++iElem){
                double jac = 0.;

                if (part_elem[iElem] == rank){
                    jac = elements_[k]->getJacobian();
                    k++;
                }

                MPI_Bcast(&jac,1,MPI_DOUBLE,part_elem[iElem],PETSC_COMM_WORLD);
                if (rank == 0) output_v << jac << std::endl;

                MPI_Barrier(PETSC_COMM_WORLD);
            }                

            if(rank == 0) output_v << "      </DataArray> " << std::endl;
        };
    

        if(rank == 0){
            output_v << "    </CellData>" << std::endl; 

            //FINALIZE OUTPUT FILE
            output_v << "  </Piece>" << std::endl;
            output_v << "  </UnstructuredGrid>" << std::endl
                     << "</VTKFile>" << std::endl;
         };

    };

return;

};
//------------------------------------------------------------------------------
//----------------------COMPUTES DRAG AND LIFT COEFFICIENTS---------------------
//------------------------------------------------------------------------------
// Metodo da classe FLuid
template<>
void Fluid<2>::dragAndLiftCoefficients(std::ofstream& dragLift){
    
    // inicializacao das variaveis
    double dragCoefficient = 0.;
    double liftCoefficient = 0.;
    double pressureDragCoefficient = 0.;
    double pressureLiftCoefficient = 0.;
    double frictionDragCoefficient = 0.;
    double frictionLiftCoefficient = 0.;
    double pitchingMomentCoefficient = 0.;
    double pMom = 0.;
    double per = 0.;
    
    // Loop principal sobre os elementos de contorno do fluido
    for (int jel = 0; jel < numBoundElems; jel++){   
        
        // reinicializacao das variaveis locais
        double dForce = 0.;
        double lForce = 0.;
        double pDForce = 0.;
        double pLForce = 0.;
        double fDForce = 0.;
        double fLForce = 0.;
        
        // Loop especifico para calculo dos coeficientes IFE
        for (int i=0; i<numberOfLines; i++){
            if (boundary_[jel] -> getBoundaryGroup() == dragAndLiftBoundary[i]){
                // verifica se o elemento de contorno atual percente ao grupo 
                //std::cout << "AQUI " << numberOfLines<< " " << i << " " << dragAndLiftBoundary[i] << std::endl;
                int iel = boundary_[jel] -> getElement(); // obtem indice do elemento 2D iel associado ao elemento 1D jel
                
                // itera sobre todos elementos para pegar o correspondente a iel
                for (int j = 0; j < numElem; ++j){
                    if (elements_[j] -> getIndex() == iel){
                        elements_[j] -> computeDragAndLiftForces(); // realiza todos os calculos
                    
                        pDForce = elements_[j] -> getPressureDragForce(); // recebe a parcela de forca do elemento
                        pLForce = elements_[j] -> getPressureLiftForce();
                        fDForce = elements_[j] -> getFrictionDragForce();
                        fLForce = elements_[j] -> getFrictionLiftForce();
                        dForce = elements_[j] -> getDragForce();
                        lForce = elements_[j] -> getLiftForce();
                        pMom += elements_[j] -> getPitchingMoment();
                        per += elements_[j] -> getPerimeter();
                    }   
                }
            };
        };
        
        // acumula para todos elementos do contorno
        pressureDragCoefficient += pDForce / 
            (0.5 * rhoInf * velocityInf[0] * velocityInf[0]); 
        pressureLiftCoefficient += pLForce / 
            (0.5 * rhoInf * velocityInf[0] * velocityInf[0]);
        
        frictionDragCoefficient += fDForce / 
            (0.5 * rhoInf * velocityInf[0] * velocityInf[0]);
        frictionLiftCoefficient += fLForce / 
            (0.5 * rhoInf * velocityInf[0] * velocityInf[0]);
        
        dragCoefficient += dForce / 
            (0.5 * rhoInf * velocityInf[0] * velocityInf[0]);
        liftCoefficient += lForce / 
            (0.5 * rhoInf * velocityInf[0] * velocityInf[0]);
        
    };

    // sincroniza os processos MPI
    MPI_Barrier(PETSC_COMM_WORLD);

    double totalDragCoefficient = 0.;
    double totalLiftCoefficient = 0.;
    double totalPressureDragCoefficient = 0.;
    double totalPressureLiftCoefficient = 0.;
    double totalFrictionDragCoefficient = 0.;
    double totalFrictionLiftCoefficient = 0.;
    double totalPitchingMomentCoefficient = 0.;;
    double perimeter = 0.;

    // MPI_Allreduce combina os valores de cada processo, reune e distrubui o valor final para todos os processos
    MPI_Allreduce(&dragCoefficient,&totalDragCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD); // soma os valores locais da variavel dragCoefficiente em total
    MPI_Allreduce(&liftCoefficient,&totalLiftCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD); // 1 is the number of elements in the buffer
    MPI_Allreduce(&pressureDragCoefficient,&totalPressureDragCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&pressureLiftCoefficient,&totalPressureLiftCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&frictionDragCoefficient,&totalFrictionDragCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&frictionLiftCoefficient,&totalFrictionLiftCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&pMom,&pitchingMomentCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(&per,&perimeter,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);

    totalPitchingMomentCoefficient = pitchingMomentCoefficient / (rhoInf * velocityInf[0] * velocityInf[0] * perimeter);

    // impressao no arquivo 
    if (rank == 0) {
        const int timeWidth = 15;
        const int numWidth = 15;
        dragLift << std::setprecision(5) << std::scientific;
        dragLift << std::left << std::setw(timeWidth) << iTimeStep * dTime;
        dragLift << std::setw(numWidth) << totalPressureDragCoefficient;
        dragLift << std::setw(numWidth) << totalPressureLiftCoefficient;
        dragLift << std::setw(numWidth) << totalFrictionDragCoefficient;
        dragLift << std::setw(numWidth) << totalFrictionLiftCoefficient;
        dragLift << std::setw(numWidth) << totalDragCoefficient;
        dragLift << std::setw(numWidth) << totalLiftCoefficient;
        dragLift << std::setw(numWidth) << totalPitchingMomentCoefficient;
        dragLift << std::endl;

        // Impressão dos resultados na janela de comando
        std::cout << "Time: " << std::setw(15) << iTimeStep * dTime;
        std::cout << "Total Pressure Drag Coefficient: " << std::setw(15) << totalPressureDragCoefficient << std::endl;
        std::cout << "Total Pressure Lift Coefficient: " << std::setw(15) << totalPressureLiftCoefficient << std::endl;
        std::cout << "Total Friction Drag Coefficient: " << std::setw(15) << totalFrictionDragCoefficient << std::endl;
        std::cout << "Total Friction Lift Coefficient: " << std::setw(15) << totalFrictionLiftCoefficient << std::endl;
        std::cout << "Total Drag Coefficient: " << std::setw(15) << totalDragCoefficient << std::endl;
        std::cout << "Total Lift Coefficient: " << std::setw(15) << totalLiftCoefficient << std::endl;
        std::cout << "Total Pitching Moment Coefficient: " << std::setw(15) << totalPitchingMomentCoefficient << std::endl;
        
    }
}

//------------------------------------------------------------------------------
//----------------------------READS FLUID INPUT FILE----------------------------
//------------------------------------------------------------------------------
template<>
void Fluid<2>::dataReading(Geometry* geometry, const std::string& inputFile, 
                           const std::string& inputMesh, const std::string& mirror,
                           const bool& deleteFiles){

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);      
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (rank == 0)std::cout << "Reading fluid data from \"" 
                            << inputFile << "\"" << std::endl;

    std::ifstream file(inputMesh);
    std::string line;
    std::getline(file, line); std::getline(file, line); std::getline(file, line); std::getline(file, line);
  
    
    geometry_ = geometry;


    //defyning the maps that are used to store the elements information
    std::unordered_map<int, std::string> gmshElement = { {1, "line"}, {2, "triangle"}, {3, "quadrilateral"}, {8, "line3"}, {9, "triangle6"}, {10, "quadrilateral9"}, {15, "vertex"}, {16, "quadrilateral8"}, {20, "triangle9"}, {21, "triangle10"}, {26, "line4"}, {36, "quadrilateral16"}, {39, "quadrilateral12"} };
    std::unordered_map<std::string, int> numNodes2 = { {"vertex", 1}, {"line", 2}, {"triangle", 3}, {"quadrilateral", 4}, {"line3", 3}, {"triangle6", 6}, {"quadrilateral8", 8}, {"quadrilateral9", 9}, {"line4", 4}, {"triangle", 9}, {"triangle10", 10}, {"quadrilateral12", 12}, {"quadrilateral16", 16}};
    std::unordered_map<std::string, std::string> supportedElements = { {"triangle", "T3"}, {"triangle6", "T6"}, {"triangle10", "T10"}, {"quadrilateral", "Q4"}, {"quadrilateral8", "Q8"}, {"quadrilateral9", "Q9"}, {"quadrilateral12", "Q12"}, {"quadrilateral16", "Q16"} };
    std::unordered_map<Line*, std::vector< std::vector<int> >> lineElements;


    //Defines input and output files    
    std::ifstream inputData(inputFile.c_str());
    std::ofstream mirrorData(mirror.c_str());

    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    
    //Read number of nodes, elements, time steps and printing frequence
    inputData >> numTimeSteps >> printFreq;
    mirrorData << "Number of Time Steps   = " << numTimeSteps << std::endl;
    mirrorData << "Printing Frequence     = " << printFreq << std::endl;
    
    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    
    //Read undisturbed velocity and pressure components
    inputData >> velocityInf[0] >> velocityInf[1] >> velocityInf[2] >> pressInf;

    mirrorData << "Undisturbed Velocity x = " << velocityInf[0] << std::endl;
    mirrorData << "Undisturbed Velocity y = " << velocityInf[1] << std::endl;
    mirrorData << "Undisturbed Velocity z = " << velocityInf[2] << std::endl;
    mirrorData << "Undisturbed Pressure   = " << pressInf << std::endl;

    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);

    //Read undisturbed density and viscosity
    inputData >> rhoInf >> viscInf;

    mirrorData << "Undisturbed Density    = " << rhoInf << std::endl;
    mirrorData << "Undisturbed Viscosity  = " << viscInf << std::endl;

    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);

    //Read time step lenght
    inputData >> dTime >> integScheme;

    mirrorData << "Time Step              = " << dTime << std::endl;
    mirrorData << "Time Integration Scheme= " << integScheme << std::endl;

    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);getline(inputData,line);

    //Read field forces
    inputData >> fieldForces[0] >> fieldForces[1] >> fieldForces[2];

    mirrorData << "Field Forces x         = " << fieldForces[0] << std::endl;
    mirrorData << "Field Forces y         = " << fieldForces[1] << std::endl;
    mirrorData << "Field Forces z         = " << fieldForces[2] << std::endl \
               << std::endl;

    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);

    //Drag and lift
    inputData >> computeDragAndLift >> numberOfLines; 
    dragAndLiftBoundary.reserve(numberOfLines);
    for (int i = 0; i < numberOfLines; ++i)
    {
        int aux;
        inputData >> aux; 
        dragAndLiftBoundary.push_back(aux);
    }
    

    mirrorData << "Compute Drag and Lift  = " << computeDragAndLift<< std::endl;
    mirrorData << "Number of Lines  = " << numberOfLines << std::endl;
    for (int i = 0; i < numberOfLines; ++i)
    {
        mirrorData << "Lines  = " << dragAndLiftBoundary[i] << std::endl;
    }



    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);

    //Printing results
    inputData >> printVelocity;              getline(inputData,line);
    inputData >> printPressure;              getline(inputData,line);
    inputData >> printVorticity;             getline(inputData,line);
    inputData >> printMeshVelocity;          getline(inputData,line);
    inputData >> printMeshDisplacement;      getline(inputData,line);
    inputData >> printJacobian;              getline(inputData,line);
    inputData >> printProcess;              

    mirrorData << "PrintVelocity              = " << printVelocity << std::endl;
    mirrorData << "PrintPressure              = " << printPressure << std::endl;
    mirrorData << "PrintVorticity             = " << printVorticity 
               << std::endl;
    mirrorData << "PrintMeshVelocity          = " << printMeshVelocity
               << std::endl;
    mirrorData << "PrintMeshDisplacement      = " << printMeshDisplacement
               << std::endl;
    mirrorData << "PrintJacobian              = " << printJacobian << std::endl;
    mirrorData << "PrintProcess               = " << printProcess << std::endl 
               << std::endl;

    getline(inputData,line);getline(inputData,line);getline(inputData,line);
    getline(inputData,line);getline(inputData,line);
 
    int dimension=2;

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++READIN MESH+++++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++PHYSICAL ENTITIES++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    int number_physical_entities;
    file >> number_physical_entities;
    std::getline(file, line);
    std::unordered_map<int, std::string> physicalEntities;
    physicalEntities.reserve(number_physical_entities);

    for (int i = 0; i < number_physical_entities; i++)
    {
        std::getline(file, line);
        std::vector<std::string> tokens = split(line, " ");
        int index;
        std::istringstream(tokens[1]) >> index;
        physicalEntities[index] = tokens[2].substr(1, tokens[2].size() - 2);
    }
    std::getline(file, line); std::getline(file, line);

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++NODES++++++++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    file >> numNodes;
    nodes_.reserve(numNodes);
    std::getline(file, line);
    int index = 0;
    if (rank == 0) std::cout << "Number of Nodes " << " " << numNodes << std::endl;
    for (int i = 0; i < numNodes; i++)
    {
        typename Node::VecLocD x;
        std::getline(file, line);
        std::vector<std::string> tokens = split(line, " ");
        bounded_vector<double,2> coord;
        std::istringstream(tokens[1]) >> x(0);
        std::istringstream(tokens[2]) >> x(1);
        //addNode(i, coord);
         Node *node = new Node(x, index++);
         nodes_.push_back(node);
    }
    std::getline(file, line); std::getline(file, line);

    mirrorData << "Nodal Coordinates " << numNodes << std::endl;
    for (int i = 0 ; i<numNodes; i++){
        typename Node::VecLocD x;
        x = nodes_[i]->getCoordinates();       
        for (int j=0; j<2; j++){
            mirrorData << x(j) << " ";
        };
        mirrorData << std::endl;
        nodes_[i] -> setVelocity(velocityInf);
        nodes_[i] -> setPreviousVelocity(velocityInf);
        double u[2];
        u[0] = 0.; u[1] = 0.;
        nodes_[i] -> setMeshVelocity(u);
    };


    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++++ELEMENTS++++++++++++++++++++++++++++++++
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    int number_elements;
    file >> number_elements;
    //elements_.reserve(number_elements);
    std::vector<Elements *>   elementsAux_;
    elementsAux_.reserve(number_elements);

    boundary_.reserve(number_elements/10);
    index = 0;
    std::getline(file, line);
    int cont = 0;

    numBoundElems = 0;
    numElem = 0;

    std::vector<BoundaryCondition*> dirichlet, neumann, glue, FSinterface;
    dirichlet = geometry_->getBoundaryCondition("DIRICHLET"); 
    neumann = geometry_->getBoundaryCondition("NEUMANN"); 
    FSinterface = geometry_->getBoundaryCondition("MOVING");
    glue = geometry_->getBoundaryCondition("GEOMETRY");


    for (int i = 0; i < number_elements; i++)
    {
        std::getline(file, line);
        std::vector<std::string> tokens = split(line, " ");
        std::vector<int> values(tokens.size(), 0);
        for (size_t j = 0; j < tokens.size(); j++)
            std::istringstream(tokens[j]) >> values[j];
        std::string elementType = gmshElement[values[1]];
        int number_nodes_per_element = numNodes2[elementType];
        std::vector<int> elementNodes;
        elementNodes.reserve(number_nodes_per_element);

        for (size_t j = 5 ; j < values.size(); j++)
            elementNodes.push_back(values[j]-1);
 
        std::string name = physicalEntities[values[3]];
        //Adding 2D elements to surfaces
        if (name[0] == 's'){

            if(rank == 0){
                if (supportedElements.find(elementType) == supportedElements.end()){
                    std::cout << elementType << " is not supported.\n";
                    exit(EXIT_FAILURE);
                }

                PlaneSurface* object = geometry_ -> getPlaneSurface(name);
                //int materialIndex = object -> getMaterial() -> getIndex();
                //double thickness = object -> getThickness();
                numElem++;

                typename Elements::Connectivity connect;
                connect.clear();
                for (int j = 0 ; j < 6; j++) connect(j) = elementNodes[j];

                Elements *el = new Elements(index++,connect,nodes_);
                elementsAux_.push_back(el);

                for (int k = 0; k<6; k++){
                    nodes_[connect(k)] -> pushInverseIncidence(index);
                };
            }
        }
        else if (name[0] == 'l')
        {
            Boundaries::BoundConnect connectB;

            connectB(0) = elementNodes[0];
            connectB(1) = elementNodes[1];
            connectB(2) = elementNodes[2];

            int ibound;

            std::string::size_type sz;   // alias of size_t
            ibound = std::stoi (&name[1],nullptr,10);

            int constrain[3];
            double value[3];

            for (int i = 0; i < dirichlet.size(); i++){
                if (name == dirichlet[i] -> getLineName()){
                     if ((dirichlet[i] -> getComponentX()).size() == 0){
                         constrain[0] = 0; value[0] = 0;
                     }else{
                         std::vector<double> c = dirichlet[i] -> getComponentX();
                         constrain[0] = 1;
                         value[0] = c[0];
                     }
                     if ((dirichlet[i] -> getComponentY()).size() == 0){
                         constrain[1] = 0; value[1] = 0;
                     }else{
                         std::vector<double> c = dirichlet[i] -> getComponentY();
                         constrain[1] = 1; // 1 dirichlet
                         value[1] = c[0];
                     }
                }
            }

            for (int i = 0; i < neumann.size(); i++){
                if (name == neumann[i] -> getLineName()){
                    if ((neumann[i] -> getComponentX()).size() == 0){
                        constrain[0] = 0; value[0] = 0;
                    }else{
                        std::vector<double> c = neumann[i] -> getComponentX();
                        constrain[0] = 0;
                        value[0] = c[0];
                    }
                    if ((neumann[i] -> getComponentY()).size() == 0){
                        constrain[1] = 0; value[1] = 0;
                    }else{
                        std::vector<double> c = neumann[i] -> getComponentX();
                        constrain[1] = 0; // 0 neumann 
                        value[1] = c[0];
                    }
                }
            }  
            
            for (int i = 0; i < glue.size(); i++){
                if (name == glue[i] -> getLineName()){
                    if ((glue[i] -> getComponentX()).size() == 0){
                        constrain[0] = 2; value[0] = 0;
                    }else{
                        std::vector<double> c = glue[i] -> getComponentX();
                        constrain[0] = 2;
                        value[0] = c[0];
                    }
                    if ((glue[i] -> getComponentY()).size() == 0){
                        constrain[1] = 2; value[1] = 0;
                    }else{
                        std::vector<double> c = glue[i] -> getComponentY();
                        constrain[1] = 2; // 2 glue ou GEOMETRY
                        value[1] = c[0];
                    }//std::cout <<"aqui " << std::endl;
                }
            }              
            for (int i = 0; i < FSinterface.size(); i++){
                if (name == FSinterface[i] -> getLineName()){
                    if ((FSinterface[i] -> getComponentX()).size() == 0){
                        constrain[0] = 3; value[0] = 0;
                    }else{
                        std::vector<double> c = FSinterface[i] -> getComponentX();
                        constrain[0] = 3;
                        value[0] = c[0];
                    }
                    if ((FSinterface[i] -> getComponentY()).size() == 0){
                        constrain[1] = 3; value[1] = 0;
                    }else{
                        std::vector<double> c = FSinterface[i] -> getComponentY();
                        constrain[1] = 3; // 3 FSinterface ou MOVING
                        value[1] = c[0];
                    }
                }
            }        
            Boundaries * bound = new Boundaries(connectB, numBoundElems++, constrain, value, ibound);
            // std::cout << "asdasd " << rank << " " << ibound << std::endl;
            boundary_.push_back(bound);           
        }   
    }


    domainDecompositionMETIS(elementsAux_);

    if (rank == 0){
        for (int i = 0; i < numElem; ++i) delete elementsAux_[i];
        elementsAux_.clear();
    }

    MPI_Barrier(PETSC_COMM_WORLD);

    std::string result;
    std::ostringstream convert;

    convert << rank+000;
    result = convert.str();
    std::string s = "mesh"+result+".dat";

    std::ifstream mesh(s.c_str(), std::ios_base::out);

    mesh >> numElem;

    elements_.reserve(numElem);

    //reading element connectivity
    for (int i = 0; i < numElem; i++){
        typename Elements::Connectivity connect;
        connect.clear();
        int ind_ = 0;

        mesh >> ind_ >> connect(0) >> connect(1) >> connect(2) >> connect(3) >> connect(4) >> connect(5);

        Elements *el = new Elements(ind_,connect,nodes_);
        elements_.push_back(el);
    };

    MPI_Barrier(PETSC_COMM_WORLD);

    MPI_Allreduce(&numElem,&numTotalElem,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);

    if (rank == 0) std::cout << "Number of elements " << number_elements << " " 
                             << numTotalElem << " " << numBoundElems << std::endl;
    mirrorData << std::endl << "Element Connectivity" << std::endl;        
    for (int jel=0; jel<numElem; jel++){
        typename Elements::Connectivity connec;
        connec=elements_[jel]->getConnectivity();       
        for (int i=0; i<4*dimension-2; i++){
            mirrorData << connec(i) << " ";
        };
        mirrorData << std::endl;
    };

    //Sets boundary constrains
    for (int ibound = 0; ibound < numBoundElems; ibound++){
        
        Boundaries::BoundConnect connectB;
        connectB = boundary_[ibound] -> getBoundaryConnectivity();
        int no1 = connectB(0);
        int no2 = connectB(1);
        int no3 = connectB(2);
        if ((boundary_[ibound] -> getConstrain(0) != 2)){
            nodes_[no1] -> setConstrainsLaplace(0,1,0);
            nodes_[no2] -> setConstrainsLaplace(0,1,0);
            nodes_[no3] -> setConstrainsLaplace(0,1,0);
        };
        if ((boundary_[ibound] -> getConstrain(1) != 2)){
            nodes_[no1] -> setConstrainsLaplace(1,1,0);
            nodes_[no2] -> setConstrainsLaplace(1,1,0);
            nodes_[no3] -> setConstrainsLaplace(1,1,0);
        };
        
        if ((boundary_[ibound] -> getConstrain(0) == 1) || (boundary_[ibound] -> getConstrain(0) == 3)){

            //Desfazer primeira parte do if para voltar a cond. cont. constante
            // if (boundary_[ibound] -> getConstrainValue(0) <= 1.){
                
            //     typename Node::VecLocD x;
            //     x = nodes_[no1]->getCoordinates();                 
                
            //     nodes_[no1] -> setConstrains(0,boundary_[ibound] -> 
            //                                  getConstrain(0),
            //                                  x(1) * boundary_[ibound] -> 
            //                                  getConstrainValue(0));
                
            //     x = nodes_[no2]->getCoordinates();                 
                
            //     nodes_[no2] -> setConstrains(0,boundary_[ibound] -> 
            //                                  getConstrain(0),
            //                                  x(1) * boundary_[ibound] -> 
            //                                  getConstrainValue(0));
                
            //     x = nodes_[no3]->getCoordinates();                 
                
            //     nodes_[no3] -> setConstrains(0,boundary_[ibound] -> 
            //                                  getConstrain(0),
            //                                  x(1) * boundary_[ibound] -> 
            //                                  getConstrainValue(0));
            //     //ate aqui
            // } else {
            nodes_[no1] -> setConstrains(0,boundary_[ibound] -> getConstrain(0),
                                     boundary_[ibound] -> getConstrainValue(0));
            nodes_[no2] -> setConstrains(0,boundary_[ibound] -> getConstrain(0),
                                     boundary_[ibound] -> getConstrainValue(0));
            nodes_[no3] -> setConstrains(0,boundary_[ibound] -> getConstrain(0),
                                     boundary_[ibound] -> getConstrainValue(0));
             // };
        };

        if((boundary_[ibound] -> getConstrain(1) == 1) || (boundary_[ibound] -> getConstrain(1) == 3)){
            nodes_[no1] -> setConstrains(1,boundary_[ibound] -> getConstrain(1),
                                     boundary_[ibound] -> getConstrainValue(1));
            nodes_[no2] -> setConstrains(1,boundary_[ibound] -> getConstrain(1),
                                     boundary_[ibound] -> getConstrainValue(1));
            nodes_[no3] -> setConstrains(1,boundary_[ibound] -> getConstrain(1),
                                     boundary_[ibound] -> getConstrainValue(1));
        };     
    };



    // For Cavity flow only
    // for (int k = 0; k < numNodes; ++k)
    // {
    //     typename Node::VecLocD x;
    //     x = nodes_[k] -> getCoordinates();
    //     if ((x(0) < 0.001) || (x(0) > 0.999))
    //     {
    //         nodes_[k] -> setConstrains(0,1.,0.);
    //         nodes_[k] -> setConstrains(1,1.,0.);
    //     }
    // }
























        //Print nodal constrains
    for (int i=0; i<numNodes; i++){

        mirrorData<< "Constrains " << i
                  << " " << nodes_[i] -> getConstrains(0) \
                  << " " << nodes_[i] -> getConstrainValue(0) \
                  << " " << nodes_[i] -> getConstrains(1) \
                  << " " << nodes_[i] -> getConstrainValue(1) << std::endl;
    }; 




    //Sets fluid elements and sides on interface boundaries
    for (int i=0; i<numBoundElems; i++){
        if ((boundary_[i] -> getConstrain(0) > 0) ||
            (boundary_[i] -> getConstrain(1) > 0)) {

            Boundaries::BoundConnect connectB;
            connectB = boundary_[i] -> getBoundaryConnectivity();

            for (int j=0; j<numElem; j++){
                typename Elements::Connectivity connect;
                connect = elements_[j] -> getConnectivity();
                
                int flag = 0;
                int side[3];
                for (int k=0; k<6; k++){
                    if ((connectB(0) == connect(k)) || 
                        (connectB(1) == connect(k)) ||
                        (connectB(2) == connect(k))){
                        side[flag] = k;
                        flag++;
                    };
                };
                if (flag == 3){
                    boundary_[i] -> setElement(elements_[j] -> getIndex());
                    //Sets element index and side
                    if ((side[0]==4) || (side[1]==4) || (side[2]==4)){
                        boundary_[i] -> setElementSide(0);
                        elements_[j] -> setElemSideInBoundary(0);
                    };
                    

                    if ((side[0]==5) || (side[1]==5) || (side[2]==5)){
                        boundary_[i] -> setElementSide(1);
                        elements_[j] -> setElemSideInBoundary(1);
                    };
                    
                    if ((side[0]==3) || (side[1]==3) || (side[2]==3)){
                        boundary_[i] -> setElementSide(2);
                        elements_[j] -> setElemSideInBoundary(2);
                    };
                };
            };
        };
    };


    //Sets Viscosity, density, time step, time integration 
    //scheme and field forces values
    for (int i=0; i<numElem; i++){
        elements_[i] -> setViscosity(viscInf);
        elements_[i] -> setDensity(rhoInf);
        elements_[i] -> setTimeStep(dTime);
        elements_[i] -> setTimeIntegrationScheme(integScheme);
        elements_[i] -> setFieldForce(fieldForces);
    };

    //Closing the file
    file.close();
    if (deleteFiles)
        system((remove2 + inputFile).c_str());

    return;
};

//------------------------------------------------------------------------------
//-------------------------SOLVE TRANSIENT FLUID PROBLEM------------------------
//------------------------------------------------------------------------------
template<>
void Fluid<2>::readInitialValues(const std::string& inputVel,const std::string& inputPres) {

    int rank;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::ifstream inputPressure(inputPres.c_str());

    std::ifstream inputVelocity(inputVel.c_str());

    std::string line;

    for (int i = 0; i < numNodes; ++i)
    {
        double u_[2];
        double uz;
        inputVelocity >> u_[0] >> u_[1] >> uz;
        //getline(inputVelocity,line);
        nodes_[i] -> setVelocity(u_);
        //if (rank==0)std::cout << "asdasd " << i << " " << u_[0] << " " << u_[1] << " " << uz << std::endl;
    }

    for (int i = 0; i < numNodes; ++i)
    {
        double a_[2];
        double p_;
        inputPressure >> a_[0] >> a_[1] >> p_;
        //getline(inputPressure,line);
        nodes_[i] -> setPressure(p_);
        //if (rank==0)std::cout << "pressure " << i << " " << a_[0] << " " << a_[1] << " " << p_ << std::endl;
    }
    

    //  std::ifstream inputData(inputFile.c_str());
    // std::ofstream mirrorData(mirror.c_str());
    // std::ifstream file(inputMesh);
    // std::string line;
    // std::getline(file, line); std::getline(file, line); std::getline(file, line); std::getline(file, line);
  


    
    

    return;
}


//------------------------------------------------------------------------------
//-------------------------SOLVE STEADY LAPLACE PROBLEM-------------------------
//------------------------------------------------------------------------------
template<>
int Fluid<2>::solveSteadyLaplaceProblem(int iterNumber, double tolerance) {

    Mat               A;
    Vec               b, u, All;
    PetscErrorCode    ierr;
    PetscInt          Istart, Iend, Ii, Ione, iterations;
    KSP               ksp;
    PC                pc;
    VecScatter        ctx;
    PetscScalar       val;
   
    int rank;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);        

    for (int inewton = 0; inewton < iterNumber; inewton++){

        ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                            2*numNodes, 2*numNodes,
                            50,NULL,50,NULL,&A); CHKERRQ(ierr);
        
        ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);
        
        //Create PETSc vectors
        ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
        ierr = VecSetSizes(b,PETSC_DECIDE,2*numNodes);CHKERRQ(ierr);
        ierr = VecSetFromOptions(b);CHKERRQ(ierr);
        ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
        ierr = VecDuplicate(b,&All);CHKERRQ(ierr);
        
        //std::cout << "Istart = " << Istart << " Iend = " << Iend << std::endl;
        
        for (int jel = 0; jel < numElem; jel++){   
            
            //if (part_elem[jel] == rank) {
            
            //Compute Element matrix
            elements_[jel] -> getSteadyLaplace();
                            
            typename Elements::LocalMatrix Ajac;
            typename Elements::LocalVector Rhs;
            typename Elements::Connectivity connec;
            
            //Gets element connectivity, jacobian and rhs 
            connec = elements_[jel] -> getConnectivity();
            Ajac = elements_[jel] -> getJacNRMatrix();
            Rhs = elements_[jel] -> getRhsVector();
            
            //Disperse local contributions into the global matrix
            //Matrix K and C
            for (int i=0; i<6; i++){
                for (int j=0; j<6; j++){

                    int dof_i = 2*connec(i);
                    int dof_j = 2*connec(j);
                    ierr = MatSetValues(A,1,&dof_i,1,&dof_j,            \
                                        &Ajac(2*i  ,2*j  ),ADD_VALUES);
                    
                    dof_i = 2*connec(i)+1;
                    dof_j = 2*connec(j);
                    ierr = MatSetValues(A,1,&dof_i,1,&dof_j,            \
                                        &Ajac(2*i+1,2*j  ),ADD_VALUES);
                    
                    dof_i = 2*connec(i);
                    dof_j = 2*connec(j)+1;
                    ierr = MatSetValues(A,1,&dof_i,1,&dof_j,            \
                                        &Ajac(2*i  ,2*j+1),ADD_VALUES);
                    
                    dof_i = 2*connec(i)+1;
                    dof_j = 2*connec(j)+1;
                        ierr = MatSetValues(A,1,&dof_i,1,&dof_j,    \
                                            &Ajac(2*i+1,2*j+1),ADD_VALUES);
                };
                                    
                //Rhs vector
            // if (fabs(Rhs(2*i  )) >= 1.e-8){
                int dofv_i = 2*connec(i);
                ierr = VecSetValues(b,1,&dofv_i,&Rhs(2*i  ),ADD_VALUES);
                //  };
                
                //if (fabs(Rhs(2*i+1)) >= 1.e-8){
                dofv_i = 2*connec(i)+1;
                ierr = VecSetValues(b,1,&dofv_i,&Rhs(2*i+1),ADD_VALUES);
                //};
            };
        };
        
        //Assemble matrices and vectors
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        
        ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
        
        //MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        //ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        
        //Create KSP context to solve the linear system
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
        
        ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
        
#if defined(PETSC_HAVE_MUMPS)
        ierr = KSPSetType(ksp,KSPPREONLY);
        ierr = KSPGetPC(ksp,&pc);
        ierr = PCSetType(pc, PCLU);
#endif
        
        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
        ierr = KSPSetUp(ksp);
        
        
        
        ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
        
        ierr = KSPGetTotalIterations(ksp, &iterations);

        //std::cout << "GMRES Iterations = " << iterations << std::endl;
        
        //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);CHKERRQ(ierr);
        
        //Gathers the solution vector to the master process
        ierr = VecScatterCreateToAll(u, &ctx, &All);CHKERRQ(ierr);
        
        ierr = VecScatterBegin(ctx, u, All, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
        
        ierr = VecScatterEnd(ctx, u, All, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
        
        ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
                
        //Updates nodal values
        double u_ [2];
        Ione = 1;

        for (int i = 0; i < numNodes; ++i){
            Ii = 2*i;
            ierr = VecGetValues(All, Ione, &Ii, &val);CHKERRQ(ierr);
            u_[0] = val;
            Ii = 2*i+1;
            ierr = VecGetValues(All, Ione, &Ii, &val);CHKERRQ(ierr);
            u_[1] = val;
            nodes_[i] -> incrementCoordinate(0,u_[0]);
            nodes_[i] -> incrementCoordinate(1,u_[1]);
        };
        
        //Computes the solution vector norm
        ierr = VecNorm(u,NORM_2,&val);CHKERRQ(ierr);
        
        if(rank == 0){
            std::cout << "MESH MOVING - ERROR = " << val 
                      << std::scientific <<  std::endl;
        };

        ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
        ierr = VecDestroy(&b); CHKERRQ(ierr);
        ierr = VecDestroy(&u); CHKERRQ(ierr);
        ierr = VecDestroy(&All); CHKERRQ(ierr);
        ierr = MatDestroy(&A); CHKERRQ(ierr);

        if(val <= tolerance){
            break;            
        };             
    };
    
    // for (int i=0; i<numElem; i++){
    //     elements_[i] -> computeNodalGradient();            
    // };

    if (rank == 0) {
        //Computing velocity divergent
        //      printResults(1);
    };

    return 0;
};

//------------------------------------------------------------------------------
//-------------------------SOLVE TRANSIENT FLUID PROBLEM------------------------
//------------------------------------------------------------------------------
template<>
int Fluid<2>::solveTransientProblem(int iterNumber, double tolerance) {

    Mat               A;
    Vec               b, u, All;
    PetscErrorCode    ierr;
    PetscInt          Istart, Iend, Ii, Ione, iterations;
    KSP               ksp;
    PC                pc;
    VecScatter        ctx;
    PetscScalar       val;
    //IS             rowperm       = NULL,colperm = NULL;
    //    MatNullSpace      nullsp;
   
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    

    std::ofstream dragLift;
    dragLift.open("dragLift.dat", std::ofstream::out | std::ofstream::app);
    if (rank == 0) {
        dragLift << "Time   Pressure Drag   Pressure Lift " 
                 << "Friction Drag  Friction Lift Drag    Lift " 
                 << std::endl;
    };    

     // Set element mesh moving parameters
    double vMax = 0., vMin = 1.e10;
    for (int i = 0; i < numElem; i++){
        double v = elements_[i] -> getJacobian();
        if (v > vMax) vMax = v;
        if (v < vMin) vMin = v;
    };
    for (int i = 0; i < numElem; i++){
        double v = elements_[i] -> getJacobian();
        double eta = 1 + (1. - vMin / vMax) / (v / vMax);
        elements_[i] -> setMeshMovingParameter(eta); // ver na tese do jefferson
    };

    ///////////////// ENTRADA SOLIDO ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// ENTRADA SOLIDO ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////ENTRADA DADOS HEADER FILLE///////////////////////////////////////////////////////////////////////////////////
    //PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

    // Obter o comunicador MPI
    MPI_Comm mpi_comm = PETSC_COMM_WORLD;
    
    //Declaracao escalares
    int nnos, nmats, npropmat, nelems, naps, nmolas, nmn, nnoscar, ndir;
    int existeCargaPermanente, existeCombinacaoNBR, tipoCombinacaoNBR, tipoanalise;
    int npassosCarga,tipoelem;
    double residuoadm, tt;

    int ncoord = 3; // Valor padrão
    int ntiposcargadin = 2;
 
    //Dinamica MEFP
    //double beta = 1.0 / 4.0; //Newmark
    //double gama = 1.0 / 2.0; //Newmark
    int limitecont=30; // Metodo Newton Raphson

    // Raio espectral: maior autovalor em modulo da matriz de amplificacao dinamica A
    double rhoinf=1.0; //rhoinf=0.0 eliminacao total altas frequencias; rhoinf=1.0 sem dissipacao (Newmark);
    double alphaM=(2.0-rhoinf)/(1.0+rhoinf); //alpha generalizado
    double alphaF=1.0/(1.0+rhoinf); //alpha generalizado
    double beta= 0.25*(1.0+alphaM-alphaF)*(1.0+alphaM-alphaF); //alpha generalizado
    double gama=0.5+alphaM-alphaF; //alpha generalizado
    alphaF=1.0; alphaM=1.0;
    //tipotensao_1(SaintVenantKirchhoff)||2(Hooke)||3(AlmansiLinear)
    int tipotensao=1;
    // Vibracao
    int resolverVibracao=0; // resolverVibracao=0 (Nao); 1 (sim) // corrigir: usar apenas PETsc; colocar laco manual para autovetores e autovalores;
    //Num_passos_tempo_direto_npassostempo(0=CalculoAutomatico_valor=Dt=tt/npassostempo)
    int npassostempo;
    //ImprimirAnaliseporPassodeCarga(0=naoImprimir_1=imprimirPosicoesCarga_2=imprimirDeslocamentosTempo_3=imprimirPosicoesTempo_4=imprimirKmolaxISE)_iapc
    int iapc=2;
    int imprimirMatrizMassa=0; //imprimirMatrizMassa=1 (SIM); 0 (NAO);
    int imprimirMatrizHd=0; //imprimirMatrizMassa=1 (SIM); 0 (NAO);
    int imprimirgj=0;
    //Tipo_Amortecimento(ksi1_Mm||ksi2_Hoz)
    double ksi1Cc = 0.0, ksi2Cc = 0.0;
    // Tipo de solver escolhido (1: LLT, 2: Cholesky, 3: Gradiente Conjugado, 4: LU, 5: inversa LU, 6: inverse Dense)
    int tipoSolver=3; 
    
    // INTEGRACAO
    int nhammer=4; // Numero de pontos de Hammer: 1,3,4,7,12 pontos; triangular 2D
    int ngauss=3; // Linhas, Quadrangular, Hexaedrica
    int ntetra=1; // Numero de pontos quadratura do tetraedro; 3D; 1,4 pontos
    
    // Malha
     int tipoMalha3D=2; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica; 
     int tipoMalha2D=1; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;
     int tipoMalha1D=0; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica; 0 para linhas;

     
     int ndirSuperficie = 2;
     int grau = 2;
     int nnoselem = 0;
     int nnoselemSuperficie=0;
     int nnoselemContorno=0;
    

     if (tipoMalha3D==2){
        for (int g=0;g<=grau;g++){
        nnoselem=0.5*(g+1)*(g+2); // nnoselem tetraedrico
        }
     } else {
        std::cout << "Elemento Nao Implementado: tipoMalha3D" << endl;
        return 1;
     }

     if (tipoMalha2D==1){
        nnoselemSuperficie = ((grau + 1) * (grau + 2)) / 2; // nnoselemSuperficie triangular

     } else {
        std::cout << "Elemento Nao Implementado: tipoMalha2D" << endl;
        return 1;
     }

     if (tipoMalha1D==0){
        nnoselemContorno = grau+1; // nnoselemContorno linhas
     } else {
        std::cout << "Elemento Nao Implementado: tipoMalha2D" << endl;
        return 1;
     }



    // Lendo os valores escalares
     nmolas =3;  nmn=0; int aplicarPesoProprio=0; int nmolasRotacao=0;
     int tipoEntradaDados=3; // tipoEntradaDados:  1 = interno programa; 2=Arquivo geral; 3=Arquivo apenas Malha; 4= header fille
     tipoanalise=1; //1_Estatica_//2_Dinamica
     tipoelem=2; //tipoelem: 1 solido 3D; 2 chapa EPT; 3 chapa EPD; 4 placa; 5 casca; 6 portico 3D; 
    npassosCarga=5;
    npassostempo=500; //Numero_passos_tempo_npt_(0_ondulatorio)_(realpositivo_transiente_Vibratorios)_npt
    residuoadm=1.0e-3;
    tt=0.04; //Tempo_total_Analise_Dinamica_tt

    if (tipoelem==1){
         ndir = 3; // 2 para 2D, 3 para 3D
    } else if(tipoelem==2 || tipoelem==3){
         ndir = 2; // 2 para 2D, 3 para 3D
         nnoselem=nnoselemSuperficie;
    } else if(tipoelem==4){

    } else if (tipoelem==5){

    } else if (tipoelem==6){

    } else {
        std::cout << "Tipo de elemento nao implementado; tipoelem: " << tipoelem << endl;
        return 5;
    }


    //////////////////ENTRADA DADOS INTERNA PROGRAMA ///////////////////////////////////////////////////////////////////////////////////////
    // Matriz de propriedades dos materiais
     nmats=1, npropmat=4;
     MatrixXd propmat = MatrixXd::Zero(nmats, npropmat);
        //    Emat(K)   vmat  ro.mat   aT
     //propmat<< 200.0e9, 0.3, 7850.0, 1.2e-5; 
     //propmat<< 3.0e7, 0.0, 0.0094116, 1.2e-5;  
     //propmat<< 1.0e4, 0.0, 1.0, 1.2e-5; 
     //propmat<< 10000, 0.0, 1.0, 1.2e-5; 
     // propmat<< 210000000000, 0.0, 1691.81, 1.2e-5;  
     propmat<< 200.0e9, 0.0, 7850.0, 1.2e-5; // arco

     MatrixXd coordno; // double float
     MatrixXi propelem,propelemSuperficie,propelemContorno; // inteiros
     int nglt,nnosSuperficie,nnosContorno, nelemsSuperficie,nelemsContorno, nelemsTotal;

     if (tipoEntradaDados==1){
    
    
    } else if (tipoEntradaDados==3) {

        // entrada apenas da malha
         //nnos=442, nelems=90;  nnosContorno=0;  nelemsSuperficie=0;  nelemsContorno=0;
           nelemsSuperficie=0;  nelemsContorno=0; //nnos=366; nelems=71; 
          
          std::vector<std::vector<int>> tempPropelemContorno;
          std::vector<std::vector<int>> tempPropelemSuperficie;
          std::vector<std::vector<int>> tempPropelem;

        std::ifstream inputFileB("entrada.txt");
        
        //std::cout << "Valor Booleano da abertura do arquivo Malha (1 esta ok, 0 falha): "; // 1 esta ok, 0 falha
        std::cout << inputFileB.is_open() <<"\n";
        if (!inputFileB.is_open()) {
           std::cerr << "Erro ao abrir o arquivo de entrada Malha." << std::endl;
           return 1;
        }

         std::string line;
         line.clear(); // limpar conteúdo da string.
          bool readingNodes = false;
          bool readingProps = false;

        while (std::getline(inputFileB, line)) {

            // Verificar se ha quebra de linha e CRLF e remover
              if (!line.empty() && line.back() == '\r') {
                 line.pop_back();
               }
            
            //line.erase(line.find_last_not_of(" \t") + 1); // Remove espaços em branco no início e no final da linha
            //line.erase(0, line.find_first_not_of(" \t"));

            // Usando a função size()
            int tamanho = line.size();
            //std::cout << "A string line tem " << tamanho << " elementos." << std::endl;

           // std::cout << "line: [" << line << "]" << std::endl;
            bool testLine = (line=="$Nodes");
           // std::cout << "testLine: " << testLine << endl;
        
        if (line == "$Nodes") {
            //std::cout << "status2.5: ok" << std::endl;
            inputFileB >> nnos;
            nglt=ndir*nnos; //int nglt=ndir*nnos+nmolasR;
            //std::cout << "nnos: " << nnos << endl;
            coordno = MatrixXd::Zero(nnos, ncoord);
            
            for (int no=0; no<nnos; ++no){
                int nodeId;
                double x, y, z;
                inputFileB >> nodeId >> x >> y >> z;
                coordno.row(nodeId - 1) << x, y, z;
                //std::cout << "status3: ok" << std::endl;
            }
            
               continue;  
            } else if (line == "$Elements") {
            //std::cout << "status4: ok" << std::endl;
             inputFileB >> nelemsTotal;
             //std::cout << "nelemsTotal: " << nelemsTotal << endl;
             for (int elemTot=0; elemTot<nelemsTotal; ++elemTot){
                int numeracao, elemTipo;
                inputFileB >> numeracao >>elemTipo;
                //std::cout << "elemTipo: " << elemTipo << endl;
                if (elemTipo==15){
                    // 15 is a element of point
                    int elemId1, no1, no2, no3;
                    inputFileB >> elemId1 >> no1 >> no2 >> no3;
                    // futuramente se necessario pode-se armazenar em matriz de pontos
                 continue;
                } else if(elemTipo==26){
                    // Elemento de linha cubico 4 nos - 3 ordem
                    int elemId1, elemId2, elemId3;
                    int no1, no2, no3, no4;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2 >> no3 >> no4;
                    nelemsContorno++;
                    tempPropelemContorno.push_back({no1, no2, no3, no4, 1}); // substituir 1 por elemId futuramente
                   
                     continue;

                } else if(elemTipo==8)  {
                    // Elemento de linha quadratico 3 nos - 2 ordem
                    int elemId1, elemId2, elemId3; // sera que tem elemId3?
                    int no1, no2, no3;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2 >> no3;
                    nelemsContorno++;
                    tempPropelemContorno.push_back({no1, no2, no3, 1}); // substituir 1 por elemId futuramente


                } else if(elemTipo==999)  {
                    // Elemento de linha linear 2 nos - 1 ordem
                    int elemId1, elemId2, elemId3; // sera que tem elemId3?
                    int no1, no2;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2;
                    nelemsContorno++;
                    tempPropelemContorno.push_back({no1, no2, 1}); // substituir 1 por elemId futuramente


                } else if(elemTipo==21){
                    // Elemento triangular 10 nos - 3 ordem
                    int elemId1, elemId2, elemId3;
                    int no1, no2, no3, no4, no5, no6, no7, no8, no9, no10;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2 >> no3 >> no4 >> no5 >> no6 >> no7 >> no8 >> no9 >> no10;
                    nelemsSuperficie++;
                    tempPropelemSuperficie.push_back({no1, no2, no3, no4, no5, no6, no7, no8, no9, no10, 1}); // substituir 1 por elemId futuramente

                } else if(elemTipo==9)  {
                    // Elemento triangular 6 nos - 2 ordem
                    int elemId1, elemId2, elemId3;
                    int no1, no2, no3, no4, no5, no6;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2 >> no3 >> no4 >> no5 >> no6;
                    nelemsSuperficie++;
                    tempPropelemSuperficie.push_back({no1, no2, no3, no4, no5, no6, 1}); // substituir 1 por elemId futuramente

                } else if(elemTipo==999)  {
                    // Elemento triangular 3 nos - 1 ordem
                    int elemId1, elemId2, elemId3;
                    int no1, no2, no3;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2 >> no3;
                    nelemsSuperficie++;
                    tempPropelemSuperficie.push_back({no1, no2, no3, 1}); // substituir 1 por elemId futuramente

                } else if(elemTipo==999) {
                    // Elemento tetraedrico 20 nos - 3 ordem
                    int elemId1, elemId2, elemId3;
                    int no1, no2, no3, no4, no5, no6, no7, no8, no9, no10, no11, no12, no13, no14, no15, no16, no17, no18, no19, no20;
                    inputFileB >> elemId1 >> elemId2 >> elemId3 >> no1 >> no2 >> no3 >> no4 >> no5 >> no6 >> no7 >> no8 >> no9 >> no10>> no11 >> no12 >> no13 >> no14 >> no15 >> no16 >> no17 >> no18 >> no19 >> no20;
                    nelems++;
                    tempPropelem.push_back({no1, no2, no3, no4, no5, no6, no7, no8, no9, no10, no11, no12, no13, no14, no15, no16, no17, no18, no19, no20, 1});
                    //propelem.resize(nelems, Eigen::NoChange);
                    //propelem.row(nelems-1) << no1, no2, no3, no4, no5, no6, no7, no8, no9, no10, no11, no12, no13, no14, no15, no16, no17, no18, no19, no20, 1; // substituir 1 por elemId futuramente

                } else {
                    std::cout << "Elemento ainda nao implementado, verifique a entrada de dados da malha." << std::endl;
                    return 99;
                }
             }
            
                continue;
            
        } else if (line == "$EndNodes") {
            continue;

        } // final if
        } //final while

        inputFileB.close(); //fecha arquivo entrada.txt da malha

            if (ndir==2){
               nnosSuperficie=nnos;
               nelems=nelemsSuperficie;
             } else if (ndir==3){
               nnosSuperficie=nnos; // calcular valor
               std::cout << "Implementar nnosSuperficie, nnosSuperficie: " << nnosSuperficie << endl;
               return 99;
              } else {
                 std::cout << "Numero de direcao nao implementada - ndir: " << ndir << endl;
              return 99;
            }

             propelemContorno = MatrixXi::Zero(nelemsContorno, nnoselemContorno + 1);
             propelemSuperficie = MatrixXi::Zero(nelemsSuperficie, nnoselemSuperficie + 1);
             propelem = MatrixXi::Zero(nelems, nnoselem + 1);

            // Transferir dados matriz Eigen - futuramente pode-se trabalhar apenas com matrizes dinamicas vector
            for (int i = 0; i < nelemsContorno; ++i) {
                for (int j = 0; j < nnoselemContorno + 1; ++j) {
                    propelemContorno(i, j) = tempPropelemContorno[i][j];
                }
            }

            for (int i = 0; i < nelemsSuperficie; ++i) {
                for (int j = 0; j < nnoselemSuperficie + 1; ++j) {
                    propelemSuperficie(i, j) = tempPropelemSuperficie[i][j];
                }
            }

            if (ndir==2){
             propelem=propelemSuperficie; //apenas ate implementar entrada separada
            } else if (ndir==3){
                 for (int i = 0; i < nelems; ++i) {
                     for (int j = 0; j < nnoselem + 1; ++j) {
                         propelem(i, j) = tempPropelem[i][j];
                     }
                 }
            }
            
            nnosContorno=0;
            //nnosContorno=nelemsContorno*nnoselemContorno; // ha nos repetidos

    } //final if geral tipoEntradaDados



    // Matriz de Restricoes Nodais
     naps=6; //19//4//16//33//10//7
     MatrixXi restrap = MatrixXi::Zero(naps, 4);
        //Apoios_(r=1_impedido||_r=0_livre||_r=2_mola)_restrap_mola
        //      NO RTX1 RTX2 RTX3
     restrap <<   1,    1,1,1,
                  3,    1,1,1,
                  69,  1,1,1,

                  2,    1,0,1,
                  4,    1,0,1,
                  70,  1,0,1;
    
    // Matriz de Mola
    int ndirMolas=3; int ndirMolasRotacao=1;
    MatrixXd mola = MatrixXd::Zero(naps, ndirMolas);
    if (nmolasRotacao>0){
        VectorXd molaR = VectorXd::Zero(nmolasRotacao);
        MatrixXi idmolaR = MatrixXi::Zero(nmolasRotacao, 2);
    }
    
     //        kX1 kX2 kX3
     //mola <<   0.0,0.0,0.0,
     //          0.0,0.0,0.0;

     //molaR<< 0.0, 0.0;
     //idmolaR<< 1, 1,
     //          10, 3;


    // Matriz de Forcas: Lineares, Superficie, Dominio
    nnoscar=nnos; //19//16//34//10//7 nnoscar=nnos; //35
    MatrixXd carganoSolido = MatrixXd::Zero(nnoscar, 3 * 4);
    VectorXi idcarganoSolido = VectorXi::Zero(nnoscar);
    /*
    //                 p    t1    t2      Qx1     Qx2    Qx3     Wx1    Wx2    Wx3      Fx1      Fx2    Fx3
    carganoSolido<< 
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,

                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,

                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,

                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,

                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,

                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,

                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0,
                    5.0e5,  0.0,  0.0,    0.0,   0.0,   0.0,      0.0,    0.0,   0.0,     0.0,      0.0,   0.0;  
                    
     idcarganoSolido<< 1,2,
                      5,6,7,8,
                      9,10,11,12,
                      13,14,15,16,
                      17,18,19,20,
                
                      21,22,23,24,
                      25,26,27,28,
                      29,30,31,32,
                      33,34,35,36,
                      37;
                      */

    
    for (int i=0; i<nnos; ++i){
        idcarganoSolido(i)=i+1;
    }
    

    // Vetor de massa adicional
     VectorXd massaadc = VectorXd::Zero(nnoscar);
     //massaadc<<0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0;
     //massaadc << 0.0;

    // Matriz campodeslo
     MatrixXd campodeslo = MatrixXd::Zero(nnoscar, 3);
     //        //  DX1 DX2 DX3
     //campodeslo<< 0.0,0.0,0.0,
     //             0.0,0.0,0.0;

    //////////////////ENTRADA DADOS INTERNA PROGRAMA ///////////////////////////////////////////////////////////////////////////////////////

     if (rank == 0) {
    // impressao dos escalares na janela de comando
    std::cout << "tipoelem: " << tipoelem << endl;
    std::cout << "nnos: " << nnos << endl;
    std::cout << "nnosSuperficie: " << nnosSuperficie << endl;
    std::cout << "nnosContorno: " << nnosContorno << endl;
    std::cout << "nmats: " << nmats << endl;
    std::cout << "nelems: " << nelems << endl;
    std::cout << "nelemsSuperficie: " << nelemsSuperficie << endl;
    std::cout << "nelemsContorno: " << nelemsContorno << endl;
    std::cout << "naps: " << naps << endl;
    std::cout << "nmolas: " << nmolas << endl;
    std::cout << "nmn: " << nmn << endl;
    std::cout << "nnoscar: " << nnoscar << endl;
    std::cout << "aplicarPesoProprio: " << aplicarPesoProprio << endl;
    std::cout << "tipoanalise: " << tipoanalise << endl;
    std::cout << "npassosCarga: " << npassosCarga << endl;
    std::cout << "npassostempo: " << npassostempo << endl;
    std::cout << "residuoadm: " << residuoadm << endl;
    std::cout << "tt: " << tt << endl;

    // impressao das matrizes na janela de comando
    std::cout << "Matriz coordno:" << endl << coordno << endl;
    std::cout << "Matriz propelem:" << endl << propelem << endl;
    std::cout << "Matriz propelemSuperficie:" << endl << propelemSuperficie << endl;
    std::cout << "Matriz propelemContorno:" << endl << propelemContorno << endl;
    std::cout << "Matriz propmat:" << endl << propmat << endl;
    std::cout << "Matriz restrap:" << endl << restrap << endl;
    std::cout << "Matriz mola:" << endl << mola << endl;
    std::cout << "Vetor idcarganoSolido:" << endl << idcarganoSolido.transpose() << endl;
    std::cout << "Matriz campodeslo:" << endl << campodeslo << endl;
    std::cout << "Matriz de massa nodais:" << endl << massaadc << endl;
    std::cout << "Matriz carganoSolido:" << endl << carganoSolido << endl;
    //inputFile.close(); //fechamento arquivo entrada de dados geral

    std::cout << "status1 - Entrada de Dados: ok\n";
     }
    //return 0;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////ENTRADA DADOS HEADER FILLE///////////////////////////////////////////////////////////////////////////////////


//////////////////////FUNCAO DE INTEGRACAO AUXILIAR ///////////////////////////////////////////////////////////////////////////////////
    // Variaveis auxiliares integracao Hammer - Triangular
    int ndirTri=2;
     MatrixXd Matrizksihammer = MatrixXd::Zero(nhammer, ndirTri);
    VectorXd Vetorwih = VectorXd::Zero(nhammer);
    if (nhammer==1){
        Matrizksihammer(0,0)=(1.0/3.0);
        Matrizksihammer(0,1)=(1.0/3.0);
        Vetorwih(0)=(1.0/2.0);
    }
    else if (nhammer==3){
       Matrizksihammer(0,0)=(1.0/6.0);
       Matrizksihammer(0,1)=(1.0/6.0);
       Matrizksihammer(1,0)=(2.0/3.0);
       Matrizksihammer(1,1)=(1.0/6.0);
       Matrizksihammer(2,0)=(1.0/6.0);
       Matrizksihammer(2,1)=(2.0/3.0);
       Vetorwih(0)=(1.0/6.0);
       Vetorwih(1)=(1.0/6.0);
       Vetorwih(2)=(1.0/6.0);
    }
    else if (nhammer==4){
       Matrizksihammer(0,0)=(1.0/3.0); Matrizksihammer(0,1)=(1.0/3.0);
       Matrizksihammer(1,0)=(1.0/5.0); Matrizksihammer(1,1)=(1.0/5.0);
       Matrizksihammer(2,0)=(3.0/5.0); Matrizksihammer(2,1)=(1.0/5.0);
       Matrizksihammer(3,0)=(1.0/5.0); Matrizksihammer(3,1)=(3.0/5.0);
       
       Vetorwih(0)=(-27.0/96.0);
       Vetorwih(1)=(25.0/96.0);
       Vetorwih(2)=(25.0/96.0);
       Vetorwih(3)=(25.0/96.0);
    }
    else if (nhammer==77){
       double auxa=0.470142064105;
       double auxb=0.101286507323;
       double auxA=0.0661970763943;
       double auxB=0.0629695902724;
       Matrizksihammer(0,0)=(1.0/3.0);          Matrizksihammer(0,1)=(1.0/3.0);
       Matrizksihammer(1,0)=(auxa);             Matrizksihammer(1,1)=(auxa);
       Matrizksihammer(2,0)=(1.0-2.0*auxa);     Matrizksihammer(2,1)=(1.0-2.0*auxa);
       Matrizksihammer(3,0)=(auxa);             Matrizksihammer(3,1)=(1.0-2.0*auxa);
       Matrizksihammer(4,0)=(auxb);             Matrizksihammer(4,1)=(auxb);
       Matrizksihammer(5,0)=(1.0-2.0*auxb);     Matrizksihammer(5,1)=(auxb);
       Matrizksihammer(6,0)=(auxb);             Matrizksihammer(6,1)=(1.0-2.0*auxb);
       
       Vetorwih(0)=(9.0/80.0);
       Vetorwih(1)=(auxA);
       Vetorwih(2)=(auxA);
       Vetorwih(3)=(auxA);
       Vetorwih(4)=(auxB);
       Vetorwih(5)=(auxB);
       Vetorwih(6)=(auxB);
    }
    else if (nhammer==7){
        //ordem Coda
       double auxa=0.470142064105;
       double auxb=0.101286507323;
       double auxA=0.0661970763943;
       double auxB=0.0629695902724;
       Matrizksihammer(0,0)=(1.0/3.0);                 Matrizksihammer(0,1)=(1.0/3.0);
       Matrizksihammer(1,0)=(0.797426985353087);       Matrizksihammer(1,1)=(0.101286507323456);
       Matrizksihammer(2,0)=(0.101286507323456);       Matrizksihammer(2,1)=(0.797426985353087);
       Matrizksihammer(3,0)=(0.101286507323456);       Matrizksihammer(3,1)=(0.101286507323456);
       Matrizksihammer(4,0)=(0.470142064105115);       Matrizksihammer(4,1)=(0.470142064105115);
       Matrizksihammer(5,0)=(0.059715871789770);       Matrizksihammer(5,1)=(0.470142064105115);
       Matrizksihammer(6,0)=(0.470142064105115);       Matrizksihammer(6,1)=(0.059715871789770);
       
       Vetorwih(0)=(0.11250); //0.1125
       Vetorwih(1)=(0.125939180544827/2.0); //
       Vetorwih(2)=(0.125939180544827/2.0);
       Vetorwih(3)=(0.125939180544827/2.0);
       Vetorwih(4)=(0.132394152788506/2.0);
       Vetorwih(5)=(0.132394152788506/2.0);
       Vetorwih(6)=(0.132394152788506/2.0);
    } else if (nhammer==12){
        Matrizksihammer(0,0)=0.501426509658179;
        Matrizksihammer(1,0)=0.249286745170910;
        Matrizksihammer(2,0)=0.249286745170910;
        Matrizksihammer(3,0)=0.873821971016996;
        Matrizksihammer(4,0)=0.063089014491502;
        Matrizksihammer(5,0)=0.063089014491502;
        Matrizksihammer(6,0)=0.053145049844816;
        Matrizksihammer(7,0)=0.310352451033785;
        Matrizksihammer(8,0)=0.636502499121399;
        Matrizksihammer(9,0)=0.310352451033785;
        Matrizksihammer(10,0)=0.636502499121399;
        Matrizksihammer(11,0)=0.053145049844816;

        Matrizksihammer(0,1)=0.249286745170910;
        Matrizksihammer(1,1)=0.249286745170910;
        Matrizksihammer(2,1)=0.501426509658179;
        Matrizksihammer(3,1)=0.063089014491502;
        Matrizksihammer(4,1)=0.063089014491502;
        Matrizksihammer(5,1)=0.873821971016996;
        Matrizksihammer(6,1)=0.310352451033785;
        Matrizksihammer(7,1)= 0.636502499121399;
        Matrizksihammer(8,1)=0.053145049844816;
        Matrizksihammer(9,1)=0.053145049844816;
        Matrizksihammer(10,1)=0.310352451033785;
        Matrizksihammer(11,1)=0.636502499121399;

        Vetorwih(0)=0.116786275726379*0.5;
        Vetorwih(1)=0.116786275726379*0.5;
        Vetorwih(2)=0.116786275726379*0.5;
        Vetorwih(3)= 0.050844906370207*0.5;
        Vetorwih(4)= 0.050844906370207*0.5;
        Vetorwih(5)= 0.050844906370207*0.5;
        Vetorwih(6)=0.082851075618374*0.5;
        Vetorwih(7)= 0.082851075618374*0.5;
        Vetorwih(8)=0.082851075618374*0.5;
        Vetorwih(9)=0.082851075618374*0.5;
        Vetorwih(10)=0.082851075618374*0.5;
        Vetorwih(11)=0.082851075618374*0.5;
    }
    // Variaveis auxiliares integracao Tetraedrica
    int ndirTetra=3;
    MatrixXd MatrizksiIntegralTetraedro = MatrixXd::Zero(ntetra, ndirTetra);
    VectorXd Vetorwitetra = VectorXd::Zero(ntetra);
    if (ntetra==1){
        MatrizksiIntegralTetraedro(0,0)=(1.0/4.0); MatrizksiIntegralTetraedro(0,1)=(1.0/4.0); MatrizksiIntegralTetraedro(0,2)=(1.0/4.0);
        
        Vetorwitetra(0)=(1.0/6.0);
    }
    else if (ntetra==4){
       int auxa=0.138196601125;
       int auxb=0.361803398875;
       MatrizksiIntegralTetraedro(0,0)=(auxa); MatrizksiIntegralTetraedro(0,1)=(auxa); MatrizksiIntegralTetraedro(0,2)=(auxa);
       MatrizksiIntegralTetraedro(1,0)=(auxa); MatrizksiIntegralTetraedro(1,1)=(auxa); MatrizksiIntegralTetraedro(1,2)=(auxb);
       MatrizksiIntegralTetraedro(2,0)=(auxa); MatrizksiIntegralTetraedro(2,1)=(auxb); MatrizksiIntegralTetraedro(2,2)=(auxa);
       MatrizksiIntegralTetraedro(3,0)=(auxb); MatrizksiIntegralTetraedro(3,1)=(auxa); MatrizksiIntegralTetraedro(3,2)=(auxa);
       
       Vetorwitetra(0)=(1.0/24.0);
       Vetorwitetra(1)=(1.0/24.0);
       Vetorwitetra(2)=(1.0/24.0);
       Vetorwitetra(3)=(1.0/24.0);
    }
    else if (ntetra==7){
       //Falta implementar 7 pontos
       double auxa=(1.0/4.0);
       double auxb=(1.0/6.0);
       double auxc=(0.5);
       MatrizksiIntegralTetraedro(0,0)=(auxa);         MatrizksiIntegralTetraedro(0,1)=(auxa);         MatrizksiIntegralTetraedro(0,2)=(auxa);
       MatrizksiIntegralTetraedro(1,0)=(auxb);         MatrizksiIntegralTetraedro(1,1)=(auxb);         MatrizksiIntegralTetraedro(1,2)=(auxb);
       MatrizksiIntegralTetraedro(2,0)=(auxb);         MatrizksiIntegralTetraedro(2,1)=(auxb);         MatrizksiIntegralTetraedro(2,2)=(auxc);
       MatrizksiIntegralTetraedro(3,0)=(auxb);         MatrizksiIntegralTetraedro(3,1)=(auxc);         MatrizksiIntegralTetraedro(3,2)=(auxb);
       MatrizksiIntegralTetraedro(4,0)=(auxc);         MatrizksiIntegralTetraedro(4,1)=(auxb);         MatrizksiIntegralTetraedro(4,2)=(auxb);

      // MatrizksiIntegralTetraedro(5,0)=(0); MatrizksiIntegralTetraedro(5,1)=(0);         MatrizksiIntegralTetraedro(5,2)=(0);
      // MatrizksiIntegralTetraedro(6,0)=(0);         MatrizksiIntegralTetraedro(6,1)=(0); MatrizksiIntegralTetraedro(6,2)=(0);
       
       Vetorwitetra(0)=(-2.0/15.0);
       Vetorwitetra(1)=(3.0/40.0);
       Vetorwitetra(2)=(3.0/40.0);
       Vetorwitetra(3)=(3.0/40.0);
       Vetorwitetra(4)=(3.0/40.0);

       //Vetorwitetra(5)=(0);
       //Vetorwitetra(6)=(0);
    }

    // Variaveis auxiliares integracao Gauss Linhas; ksi de 0 a 1, conforme convencao de Pascon
    int ndirLinha=1;
    MatrixXd Matrizgauss = MatrixXd::Zero(ngauss, ndirLinha);
    VectorXd Vetorwig = VectorXd::Zero(ngauss);
    if (ngauss==1){
        Matrizgauss(0,0)=(0.5);
        Vetorwig(0)=(1.0);
    }
    else if (ngauss==2){
       Matrizgauss(0,0)=(0.211324865405187);
       Matrizgauss(1,0)=(0.788675134594813);
       Vetorwig(0)=(0.5);
       Vetorwig(1)=(0.5);
    }
    else if (ngauss==3){
       Matrizgauss(0,0)=(0.112701665379258);
       Matrizgauss(1,0)=(0.5);
       Matrizgauss(2,0)=(0.887298334620742);

       Vetorwig(0)=(0.277777777777776);
       Vetorwig(1)=(0.444444444444444);
       Vetorwig(2)=(0.277777777777776);
    } else if (ngauss==4){
       Matrizgauss(0,0)=(0.069431844202974);
       Matrizgauss(1,0)=(0.330009478207572);
       Matrizgauss(2,0)=(0.669990521792428);
       Matrizgauss(3,0)=(0.930568155797026);

       Vetorwig(0)=(0.173927422568724);
       Vetorwig(1)=(0.326072577431273);
       Vetorwig(2)=(0.326072577431273);
       Vetorwig(3)=(0.173927422568724);
    } else if (ngauss==5){
       Matrizgauss(0,0)=(0.046910077030668);
       Matrizgauss(1,0)=(0.230765344947158);
       Matrizgauss(2,0)=(0.5);
       Matrizgauss(3,0)=(0.769234655052841);
       Matrizgauss(4,0)=(0.953089922969332);

       Vetorwig(0)=(0.118463442528091);
       Vetorwig(1)=(0.239314335249683);
       Vetorwig(2)=(0.284444444444444);
       Vetorwig(3)=(0.239314335249683);
       Vetorwig(4)=(0.118463442528091);
    } else if (ngauss==6){
       Matrizgauss(0,0)=(0.033765242898424);
       Matrizgauss(1,0)=(0.169395306766868);
       Matrizgauss(2,0)=(0.380690406958402);
       Matrizgauss(3,0)=(0.619309593041598);
       Matrizgauss(4,0)=(0.830604693233132);
       Matrizgauss(5,0)=(0.966234757101576);

       Vetorwig(0)=(0.085662246189581);
       Vetorwig(1)=(0.180380786524069);
       Vetorwig(2)=(0.233956967286345);
       Vetorwig(3)=(0.233956967286345);
       Vetorwig(4)=(0.180380786524069);
       Vetorwig(5)=(0.085662246189581);
        
    } else if (ngauss==7){
       Matrizgauss(0,0)=(0.025446043828621);
       Matrizgauss(1,0)=(0.129234407200303);
       Matrizgauss(2,0)=(0.297077424311301);
       Matrizgauss(3,0)=(0.5);
       Matrizgauss(4,0)=(0.702922575688699);
       Matrizgauss(5,0)=(0.870765592799697);
       Matrizgauss(6,0)=(0.974553956171379);

       Vetorwig(0)=(0.064742483084431);
       Vetorwig(1)=(0.139852695744638);
       Vetorwig(2)=(0.190915025252560);
       Vetorwig(3)=(0.208979591836735);
       Vetorwig(4)=(0.190915025252560);
       Vetorwig(5)=(0.139852695744638);
       Vetorwig(6)=(0.064742483084431);
        
    } else if (ngauss==8){
       Matrizgauss(0,0)=(0.019855071751232);
       Matrizgauss(1,0)=(0.101666761293187);
       Matrizgauss(2,0)=(0.237233795041836);
       Matrizgauss(3,0)=(0.408282678752175);
       Matrizgauss(4,0)=(0.591717321247825);
       Matrizgauss(5,0)=(0.762766204958164);
       Matrizgauss(6,0)=(0.898333238706813);
       Matrizgauss(7,0)=(0.980144928248768);

       Vetorwig(0)=(0.050614268145185);
       Vetorwig(1)=(0.111190517226687);
       Vetorwig(2)=(0.156853322938944);
       Vetorwig(3)=(0.181341891689181);
       Vetorwig(4)=(0.181341891689181);
       Vetorwig(5)=(0.156853322938944);
       Vetorwig(6)=(0.111190517226687);
       Vetorwig(7)=(0.050614268145185);
        
    } else if (ngauss==9){
        
    } else if (ngauss==10){
        
    } else if (ngauss==11){
        
    } else if (ngauss==12){
        
    }

    // Variaveis auxiliares integracao Gauss Quadrangular

    // Variaveis auxiliares integracao Gauss Hexaedrico

  
     //////////////////////FUNCAO DE INTEGRACAO/////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    

    //////////////////////FUNCAO DE MONTAGEM DOS COEFICIENTES DAS FUNCOES DE FORMA DE ORDEM QUALQUER ///////////////////////////////////////////////////
        Eigen::MatrixXd coef = Eigen::MatrixXd::Zero(nnoselemSuperficie,nnoselemSuperficie); // retorna
        Eigen::MatrixXd dcoef_dksi1 = Eigen::MatrixXd::Zero(nnoselemSuperficie,nnoselemSuperficie); // retorna
        Eigen::MatrixXd dcoef_dksi2 = Eigen::MatrixXd::Zero(nnoselemSuperficie,nnoselemSuperficie); // retorna

        funcaoCoeficientesTriangular(coef, dcoef_dksi1, dcoef_dksi2, grau); // a funcao retorna coef,dcoef_ksi calculados ordem qualquer

    //////////////////////FUNCAO DE MONTAGEM DOS COEFICIENTES DAS FUNCOES DE FORMA DE ORDEM QUALQUER ///////////////////////////////////////////////////

    
    
    //////////////////////FUNCAO GLOBAL ANALISE DINAMICA MEFP/////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////FUNCAO GLOBAL ANALISE DINAMICA MEFP/////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Configuracao Cabecalho global Arq Programador - dados para depuracao do codigo
    std::ofstream outputFile("saida_dados.txt"); //abertura do arquivo de saida de dados
    if (!outputFile.is_open()) {
        std::cerr << "Erro ao abrir o arquivo de saida de dados." << std::endl;
        return 1; // saida do programa com codigo de erro
    }
    if (rank==0){
        outputFile << "'**** => Analise Dinamica por Metodo dos Elementos Finitos Posicional.\n:" << std::endl;
        outputFile << "'\n********************************************************************************\n" << std::endl;
    }
    
    // Inicialize matrizes PETSc
    Mat Hoz, Hd, Mm, Mp, Cc;
    MatCreateSeqAIJ(PETSC_COMM_SELF, nglt, nglt, PETSC_DECIDE, PETSC_NULL, &Hoz);
    MatCreateSeqAIJ(PETSC_COMM_SELF, nglt, nglt, PETSC_DECIDE, PETSC_NULL, &Hd);
    MatCreateSeqAIJ(PETSC_COMM_SELF, nglt, nglt, PETSC_DECIDE, PETSC_NULL, &Mm);
    MatCreateSeqAIJ(PETSC_COMM_SELF, nglt, nglt, PETSC_DECIDE, PETSC_NULL, &Mp);
    MatCreateSeqAIJ(PETSC_COMM_SELF, nglt, nglt, PETSC_DECIDE, PETSC_NULL, &Cc);

    // Montar as matrizes
     MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
     
     MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY);
     
     MatAssemblyBegin(Mm, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Mm, MAT_FINAL_ASSEMBLY);
     
     MatAssemblyBegin(Mp, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Mp, MAT_FINAL_ASSEMBLY);
     
     MatAssemblyBegin(Cc, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Cc, MAT_FINAL_ASSEMBLY);
    // MatCreate(mpi_comm, &Hoz);
    // MatSetSizes(Hoz, PETSC_DECIDE, PETSC_DECIDE, nglt, nglt);
    // MatSetType(Hoz, MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Hoz, PETSC_DECIDE, PETSC_NULL, PETSC_DECIDE, PETSC_NULL);
    // MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
// 
    // MatCreate(mpi_comm, &Hd);
    // MatSetSizes(Hd, PETSC_DECIDE, PETSC_DECIDE, nglt, nglt);
    // MatSetType(Hd, MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Hd, PETSC_DECIDE, PETSC_NULL, PETSC_DECIDE, PETSC_NULL);
    // MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY);
// 
    // MatCreate(mpi_comm, &Mm);
    // MatSetSizes(Mm, PETSC_DECIDE, PETSC_DECIDE, nglt, nglt);
    // MatSetType(Mm, MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Mm, PETSC_DECIDE, PETSC_NULL, PETSC_DECIDE, PETSC_NULL);
    // MatAssemblyBegin(Mm, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(Mm, MAT_FINAL_ASSEMBLY);
// 
    // MatCreate(mpi_comm, &Mp);
    // MatSetSizes(Mp, PETSC_DECIDE, PETSC_DECIDE, nglt, nglt);
    // MatSetType(Mp, MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Mp, PETSC_DECIDE, PETSC_NULL, PETSC_DECIDE, PETSC_NULL);
    // MatAssemblyBegin(Mp, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(Mp, MAT_FINAL_ASSEMBLY);
// 
    // MatCreate(mpi_comm, &Cc);
    // MatSetSizes(Cc, PETSC_DECIDE, PETSC_DECIDE, nglt, nglt);
    // MatSetType(Cc, MATMPIAIJ);
    // MatMPIAIJSetPreallocation(Cc, PETSC_DECIDE, PETSC_NULL, PETSC_DECIDE, PETSC_NULL);
    // MatAssemblyBegin(Cc, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(Cc, MAT_FINAL_ASSEMBLY);

    std::vector<PetscInt> rows;
    std::vector<PetscInt> cols;
    std::vector<PetscScalar> values;

    Eigen::MatrixXd MmLocal = Eigen::MatrixXd::Zero(ndir*nnoselemSuperficie,ndir*nnoselemSuperficie); //apagar
   
       // Declaração dos vetores PETSc
       Vec Vjo, Ajo, Vjf, Ajf, Qs, Rs, Pn, Ujrecal, Yjo, Yjf, Fjint, Fjext, Ys, Vs, As;
       /*
       // Criação dos vetores PETSc
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Vjo);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Ajo);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Vjf);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Ajf);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Qs);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Rs);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Pn);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Ujrecal);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Yjo);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Yjf);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Fjint);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Fjext);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Ys);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &Vs);
       VecCreateMPI(PETSC_COMM_WORLD, nglt, PETSC_DETERMINE, &As);
       */

       VecCreateSeq(PETSC_COMM_SELF, nglt, &Vjo);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Ajo);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Vjf);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Ajf);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Qs);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Rs);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Pn);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Ujrecal);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Yjo);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Yjf);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Fjint);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Fjext);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Ys);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &Vs);
       VecCreateSeq(PETSC_COMM_SELF, nglt, &As);
       
       // VecSetOption para indicar que as entradas fora do processo (off-process entries) podem ser ignoradas, tornando a inicialização mais eficiente. 
       VecSetOption(Vjo, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Ajo, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Vjf, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Ajf, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Qs, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Rs, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Pn, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Ujrecal, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Yjo, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Yjf, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Fjint, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Fjext, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Ys, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(Vs, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       VecSetOption(As, VEC_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
       
       // Zerando todas as entradas
       VecSet(Vjo, 0.0);
       VecSet(Ajo, 0.0);
       VecSet(Vjf, 0.0);
       VecSet(Ajf, 0.0);
       VecSet(Qs, 0.0);
       VecSet(Rs, 0.0);
       VecSet(Pn, 0.0);
       VecSet(Ujrecal, 0.0);
       VecSet(Yjo, 0.0);
       VecSet(Yjf, 0.0);
       VecSet(Fjint, 0.0);
       VecSet(Fjext, 0.0);
    
    double tol = residuoadm;
    double t = 0.0, lok = 0.0,Lokmin = 1000.0,Lokmax = 0.0;
    int elLokmin,elLokmax;
    double Kmin,romin,Kmax,romax; 
    double wihammer=0.0, wigauss=0.0, witetra=0.0;
    
    int no = 0;
    int j = 0;
    int cont = 0;
    //double Dt = tt / npassostempo;
    npassostempo=0; //conferir
    double Dt=0.0; // conferir
     tt=0.0;
    
    //Calculo dos valores iniciais - dir=0,1,2 -> 1,2,3
     for (int no = 0; no < nnos; ++no) {
        for (int dir = 0; dir < ndir; ++dir) {
            double Yaux = coordno(no, dir);
            int j = ndir * (no) + dir;
            //Yjo(j) += Yaux;
            VecSetValue(Yjo, j, Yaux, ADD_VALUES);
        }
    }
    // Montagem final do vetor Yjo
     VecAssemblyBegin(Yjo);
     VecAssemblyEnd(Yjo);
    
    //Yjf = Yjo;
    VecCopy(Yjo, Yjf);// Copia vetor Yjo para Yjf

    // Recalque nos carregos campodeslo (adapte conforme necessário)
    // double recalque = campodeslo(pos1);
    // for (int no = 0; no < nnos; ++no) {
    //     Yjf.segment(dim * no, dim) += recalque;
    // }
    
    //Primeira Velocidade atual
    //Vjf = Vjo;
    VecCopy(Vjo, Vjf);
    
    
     

    //////////////// ENTRADA SOLIDO e INICIALIZACAO ACIMA ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////// ENTRADA SOLIDO e INICIALIZACAO ACIMA ///////////////////////////////////////////////////////////////////////////////////////////////////////////////



     //////////////////////LACO GLOBAL TEMPORAL ////////////////////////////////////////////////////////////////////////////////////////////  
     //////////////////////LACO GLOBAL TEMPORAL ////////////////////////////////////////////////////////////////////////////////////////////       
    iTimeStep = 0;

    double dTimeAux=dTime;
    
    npassostempo=numTimeSteps; //conferir
    

    for (iTimeStep = 0; iTimeStep < numTimeSteps; iTimeStep++){

        for (int i = 0; i < numElem; i++)
            elements_[i] -> getParameterSUPG();
    
        //Start the analysis with first order time integration and then change to the user defined
        
        if (integScheme < 1.){
             double one = 1.;
            if (iTimeStep == 0){
               
                for (int i = 0; i < numElem; i++){
                    elements_[i] -> setTimeIntegrationScheme(one);
                    dTime = dTimeAux;//*10.; //passo de tempo tempor�rio ?
                    elements_[i] -> setTimeStep(dTime);
                };
            };
            if (iTimeStep == 5) {
                for (int i = 0; i < numElem; i++){
                    elements_[i] -> setTimeIntegrationScheme(integScheme);
                    dTime = dTimeAux; //passo de tempo permanente ?
                    elements_[i] -> setTimeStep(dTime);
                };
            };
             
        };
        
        // passos de tempo no solido
        Dt=dTime; // conferir
        tt=Dt*npassostempo;

        //set different  iterationNumbers 
        int iterNumber2=iterNumber;
        if(iTimeStep < 4)iterNumber2 = 10;


        if (rank == 0) {std::cout << "------------------------- TIME STEP = "
                                  << iTimeStep << " -------------------------"
                                  << std::endl;}
        
        for (int i = 0; i < numNodes; i++){
            double accel[2], u[2], uprev[2];
            
            //Compute acceleration
            u[0] = nodes_[i] -> getVelocity(0);
            u[1] = nodes_[i] -> getVelocity(1);
            
            uprev[0] = nodes_[i] -> getPreviousVelocity(0);
            uprev[1] = nodes_[i] -> getPreviousVelocity(1); // valores do passo anterior
            
            accel[0] = (u[0] - uprev[0]) / dTime;
            accel[1] = (u[1] - uprev[1]) / dTime; // colocar depois solve Laplace?
            
            nodes_[i] -> setAcceleration(accel);
            
            //Updates velocity
            nodes_[i] -> setPreviousVelocity(u);

        };

        // Moving boundary_ valores do passo anterior: ok
        for (int i = 0; i < numNodes; i++){
            typename Node::VecLocD x;
            
            x = nodes_[i] -> getCoordinates();
            nodes_[i] -> setPreviousCoordinates(0,x(0));
            nodes_[i] -> setPreviousCoordinates(1,x(1)); // valores do passo anterior
        };
       

        for (int i=0; i< numNodes; i++){
            typename Node::VecLocD x,xp;
            double u[2];
            
            x = nodes_[i] -> getCoordinates();
            xp = nodes_[i] -> getPreviousCoordinates(); // valores do passo anterior
            
            u[0] = (x(0) - xp(0)) / dTime;
            u[1] = (x(1) - xp(1)) / dTime; // colocar depois solve Laplace?

            nodes_[i] -> setMeshVelocity(u);
        };

         double duNorm=100.;



      /////////////////////// PARTE DO SOLIDO ////////////////////////////////////////////////////////////////////////////////
//////////////////////LACO GLOBAL TEMPORAL ////////////////////////////////////////////////////////////////////////////////////////////
    //LACO GLOBAL TEMPORAL - METODO NEWMARK-Alpha Generalizado
    //for (int passotempo = 1; passotempo <= npassostempo; ++passotempo) {
           int passotempo=iTimeStep+1;
            t += Dt; //verificar
            
            // std::cout << " ***************** PassoTempo:" << std::endl << passotempo << " ***************** " << std::endl;

            // Atualização de variáveis de tempo passado
            //Eigen::VectorXd Ys = Yjf; Eigen::VectorXd Vs = Vjf; Eigen::VectorXd As = Ajf;
            VecCopy(Yjf, Ys); VecCopy(Vjf, Vs); VecCopy(Ajf, As);

            // Cálculo de Qs e Rs a partir do segundo tempo
            if (passotempo != 1) {
                //Qs = Yjf / (beta * (Dt * Dt)) + Vjf / (beta * Dt) + ((1.0 / (2.0 * beta)) - 1.0) * Ajf;
                // Calculo de Qs
                 VecCopy(Yjf, Qs);
                 VecScale(Qs, 1.0 / (beta * Dt * Dt));
                 VecAXPY(Qs, 1.0 / (beta * Dt), Vjf);
                 VecAXPY(Qs, ((1.0 / (2.0 * beta)) - 1.0), Ajf);
                
                // Calculo Rs
                 //Rs = Vjf + Dt * (1.0 - gama) * Ajf;
                 VecCopy(Vjf, Rs);
                 VecAXPY(Rs, Dt * (1.0 - gama),Ajf);

                // Calculo Pn
                 //Pn=(1.0-alphaF)*(Fjint-Fjext)+(1.0-alphaM)*Mm*As+(1.0-alphaF)*Cc*Vs+(-alphaM*Mm-alphaF*gama*Dt*Cc)*Qs+alphaF*Cc*Rs;
                 VecCopy(Fjint, Pn);
                 VecAXPY(Pn, -1.0,Fjext);
                 VecScale(Pn, (1.0-alphaF));

                 Mat MmCc;
                 MatDuplicate(Mm, MAT_COPY_VALUES, &MmCc);
                 MatScale(MmCc, -alphaM);
                 MatAXPY(MmCc, -alphaF*gama*Dt, Cc, DIFFERENT_NONZERO_PATTERN);

                 Vec MmAs, CcVs, MmCcQs, CcRs;
                 VecDuplicate(As, &MmAs); VecDuplicate(As, &CcVs); VecDuplicate(As, &MmCcQs); VecDuplicate(As, &CcRs);

                 MatMult(Mm, As, MmAs);
                 MatMult(Cc, Vs, CcVs);
                 MatMult(MmCc, Qs, MmCcQs);
                 MatMult(Cc, Rs, CcRs);

                 VecAXPY(Pn, (1.0-alphaM),MmAs);
                 VecAXPY(Pn, (1.0-alphaF), CcVs);
                 VecAXPY(Pn, 1.0, MmCcQs);
                 VecAXPY(Pn, alphaF, CcRs);

                  // Liberacao de memoria
                  MatDestroy(&MmCc);
                  VecDestroy(&MmAs); 
                  VecDestroy(&CcVs); 
                  VecDestroy(&MmCcQs); 
                  VecDestroy(&CcRs); 
            }           

    //////////////////////FORCAS EXTERNAS ////////////////////////////////////////////////////////////////////////////////////////////
    //Calculo do vetor global total forcas externas tempo t

    // Forcas externas Pontuais (nodais diretas)
    //Eigen::VectorXd Fjext = Eigen::VectorXd::Zero(nglt);
    //Eigen::VectorXd Ujrecal = Eigen::VectorXd::Zero(nglt);
    VecSet(Fjext, 0.0);
    //VecSet(Ujrecal, 0.0);
    for (int m = 0; m < nnoscar; ++m) {
        int nocar = idcarganoSolido(m);
        for (int dir = 0; dir < ndir; ++dir) {
            int posaux = ndir * (nocar - 1) + dir;
            //Fjext(posaux) = Fjext(posaux)+cargano(m, dir)+cargano(m, ndir+dir) * t+cargano(m,2*ndir+ dir) * std::cos(t);
            //Fjext(posaux) += carganoSolido(m, 3*3+dir);
            //Ujrecal(posaux) += campodeslo(m, dir);
            PetscScalar carga = carganoSolido(m, 3*3 + dir);
            //PetscScalar deslocamento = campodeslo(m, dir);

            // Adiciona a carga ao vetor Fjext
             VecSetValue(Fjext, posaux, carga, ADD_VALUES);

            // Adiciona o deslocamento ao vetor Ujrecal
             //VecSetValue(Ujrecal, posaux, deslocamento, ADD_VALUES);
        }
    }

    // Finaliza a montagem dos vetores
     VecAssemblyBegin(Fjext);
     VecAssemblyEnd(Fjext);
     VecAssemblyBegin(Ujrecal);
     VecAssemblyEnd(Ujrecal);


    // Montagem e Integracao da forca externa de dominio 3D
    // Forcas de Volume - Dominio - 3D - Integracao Tetraedrica
    if (ndir==3){
    for (int el = 0; el < nelems; ++el) {
      // Dado um elemento conhece as Matrizez nodais: 1 Xil do dominio, uma Xil para cada superficie do elemento
      Eigen::MatrixXd Xil = Eigen::MatrixXd::Zero(nnoselem,ndir); // Vale apenas para o dominio
      for (int i = 0; i < ndir; ++i) {
      for (int y = 0; y < nnoselem; ++y) {
        int noy=propelem(el, y);  // Pega da Matriz de conectividade do Dominio
        Xil(y,i)+=coordno(noy-1,i);  // Xil -> Xiy, dir i, noelem y; Matriz coordenadas dominio do elemento 3D
       }
      }

    
     int tipoFuncaoForma=2; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;

     Eigen::VectorXd ksi = Eigen::VectorXd::Zero(ndir);
     Eigen::VectorXd phi = Eigen::VectorXd::Zero(nnoselem);
     Eigen::MatrixXd dphi_dksi = Eigen::MatrixXd::Zero(nnoselem,ndir);
     Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(ndir,ndir);
     Eigen::MatrixXd inversaA0 = Eigen::MatrixXd::Zero(ndir,ndir);

    for (int itetra = 0; itetra < ntetra; ++itetra) {
    for (int a = 0; a < nnoselem; ++a) {
    for (int i = 0; i < ndir; ++i) {
    for (int y = 0; y < nnoselem; ++y) {
      // Forca Dominio: elemento Tetraedro - Hexaedro - Prismatico
    





















        } //final noelem y, no do tetraedro
       } //final dir i, dir do tetraedro
      } // final noelem a, no de carga
     } // final itetra integracao tetraedrico
   
    } // Final for el dominio
    } // final if verificando se eh elemento 3D
   


    // Forcas de Superficie - Area: for nelemsSuperficie
    for (int el=0; el< nelemsSuperficie; el++){
     // Assim temos certeza que o elemento pertence a superficie 2D - Integracao Hammer
   
      Eigen::MatrixXd Xil = Eigen::MatrixXd::Zero(nnoselemSuperficie,ndirSuperficie); // Vale apenas para superficie 2D
      for (int i = 0; i < ndirSuperficie; ++i) {
      for (int y = 0; y < nnoselemSuperficie; ++y) {
        int noy=propelemSuperficie(el, y);  // Pega da Matriz de conectividade da Superficie
        Xil(y,i)+=coordno(noy-1,i);  // Xil -> Xiy, dir i, noelem y; Matriz coordenadas superficie do elemento 2D
       }
      }
     int tipoFuncaoForma=1; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;

     

    for (int ihammer = 0; ihammer < nhammer; ++ihammer) {

        Eigen::VectorXd phi = Eigen::VectorXd::Zero(nnoselemSuperficie);
        Eigen::MatrixXd dphi_dksi = Eigen::MatrixXd::Zero(nnoselemSuperficie,ndirSuperficie);
        Eigen::VectorXd ksi = Eigen::VectorXd::Zero(ndirSuperficie);
        Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);

        double ksi1 = Matrizksihammer(ihammer,0);
        double ksi2 = Matrizksihammer(ihammer,1);
        double ksi3 = 0.0;
    
        ksi(0)=ksi1;
        ksi(1)=ksi2;
        if (tipoFuncaoForma==2 ||  tipoFuncaoForma==4 || tipoFuncaoForma==5){
          ksi(2)=ksi3; //se tiver 3 direcoes; ou seja solido 3D
        }

        //FuncaoFormaTriangulo(tipoFuncaoForma, ndirSuperficie,  grau,  ksi, phi, dphi_dksi); 
        FuncaoFormaTriangulo(coef, dcoef_dksi1, dcoef_dksi2, grau, ksi,  phi,  dphi_dksi); // a funcao retorna phi e dphi_dksi calculados
 
        // Propriedades Geometricas Iniciais: calculadas no ponto de hammer
    
        // Matriz Gradiente da Funcao Mudanca de Configuracao Inicial A0
        //A0=dphi_dksi.transpose()*Xil; 
        // Matriz Gradiente da Funcao Mudanca de Configuracao Inicial A0
          A0=Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
          for (int i=0;i<ndirSuperficie;++i){
          for (int j=0;j<ndirSuperficie;++j){
            for (int no=0;no<nnoselemSuperficie;++no){
                A0(i,j)+=dphi_dksi(no,j)*Xil(no,i);

             }
          }
          }
          //std::cout << "A0 novo:\n" << A0 << std::endl;



        //std::cout << "el:\n" << el << std::endl; std::cout << "ksi:\n" << ksi << std::endl; std::cout << "dphi_dksi:\n" << dphi_dksi << std::endl;
        //std::cout << "Xil:\n" << Xil << std::endl; std::cout << "A0:\n" << A0 << std::endl;
        
    
        //Jacobiano de A0 - Determinante da Matriz Mudanca de Configuracao Inicial
        double J0 = A0.determinant();   
        //std::cout << "J0:\n" << J0 << std::endl; 

         if (J0 < 10e-8) {
         std::cout << "Mapeamento Inicial Problematica, pois J0 do elemento de Superficie: " << el << " nulo ou menor que 10e-8." << std::endl;
         std::cout << "Verifique Forcas de Superficie." << std::endl;
         return 1;
        }

       //Peso de hammer
       wihammer=Vetorwih(ihammer);

    for (int a = 0; a < nnoselemSuperficie; ++a) {
    for (int i = 0; i < ndirSuperficie; ++i) {
    for (int y = 0; y < nnoselemSuperficie; ++y) {

    // Forca Superficie: elemento Triangular ou Quadrangular
    
    // Calculo phigama
     double phigama=phi(y);

    // Calculo phialpha
      double phialpha=phi(a);
    
    // Calculo do valor da carga de superficie nodal; a eh o noelem do carregamento
     double Qia=0.0;
     for (int idcar=0; idcar<nnoscar;idcar++){
        int noa=propelemSuperficie(el,a); // no de carregamento na integracao
        int noidcar=idcarganoSolido(idcar); // no realmente carregado
        if (noa==noidcar){
          Qia=carganoSolido(idcar,3+i); // realmente eh idcar, pois carganoSolido tem tamanho nnoscar
        }
     }
       
    // Integracao e Montagem vetor de forca externa
        int noy=propelemSuperficie(el, y);  // Pega da Matriz de conectividade
        int pos=ndir*(noy-1)+i;
        //Fjext(pos)+= phigama*phialpha*wihammer*J0*Qia;
        PetscScalar valor = phigama*phialpha*wihammer*J0*Qia;
        VecSetValue(Fjext, pos, valor, ADD_VALUES);

        } //final for y nnoselem
       } // final for i ndir
      } // final for a nnoselem
     } //final for ihammer nhammer
    } //for el Superficie

    // Finaliza a montagem do vetor
     VecAssemblyBegin(Fjext);
     VecAssemblyEnd(Fjext);




    // Forcas de Linhas - Contorno em Curva - 1D - Integracao Gauss
    // pegar identificador da matriz de carganoSolidos

    for (int el=0; el<nelemsContorno; ++el){
        int no1=propelemContorno(el,0);
        int no2=propelemContorno(el,1);
        int no3=propelemContorno(el,2);
        int no4=propelemContorno(el,3);

    // verificar se o elemento esta carregado
    int contIf=0;
    for (int i_car=0; i_car<nnoscar; ++i_car){
        int nocar=idcarganoSolido(i_car);
        contIf += (nocar == no1) + (nocar == no2) + (nocar == no3) + (nocar == no4);
        } // final for nocar


    if (contIf >= 2){
                
     int tipoFuncaoForma=0; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;



     for (int igauss = 0; igauss < ngauss; ++igauss) {
         Eigen::VectorXd phi = Eigen::VectorXd::Zero(nnoselemContorno);
         Eigen::MatrixXd dphi_dksi = Eigen::MatrixXd::Zero(nnoselemContorno,1);
         Eigen::VectorXd ksi = Eigen::VectorXd::Zero(1);
   
         wigauss= Vetorwig(igauss);
         double ksi1 = Matrizgauss(igauss,0);
         ksi(0)=ksi1;

         FuncaoFormaLinha(tipoFuncaoForma, 1,  grau,  ksi, phi, dphi_dksi); // a funcao retorna phi e dphi_dksi calculados

         // Calculo do J0Linha
         double termoaux=0, J0Linha=0;
         double Vx, Vy, vx, vy;
        for (int i = 0; i < ndir; ++i) {
        double termoaux=0;
        for (int y = 0; y < nnoselemContorno; ++y) {
            //Acessar Xiy do elemento no contorno -> propelemContorno
            int noy =propelemContorno(el,y);
            //std::cout << "noy:" << std::endl << noy << std::endl;
            // Xil(y,i)+=coordno(noy-1,i);  // Xil -> Xiy, dir i, noelem y; Matriz coordenadas superficie do elemento 2D
            double xil=coordno(noy-1,i); 
            termoaux+=dphi_dksi(y,0)*xil;
            //std::cout << "xil:" << std::endl << xil << std::endl;
            //std::cout << "termoaux:" << std::endl << termoaux << std::endl;

            } // Final for noelem y
            J0Linha+=pow(termoaux, 2);
            if (i==0){
                Vx=termoaux;
            } else if (i==1){
                Vy= termoaux;
            }
            //std::cout << "termoaux componente:" << std::endl << termoaux << std::endl;
            //std::cout << "J0Linha:" << std::endl << J0Linha << std::endl;
        } // final for dir i

        // Jacobiano de uma curva
        J0Linha=pow(J0Linha, 0.5);
        if (J0Linha<1.0e-16){
            std::cout << "J0Linha = dx/dksi, J0Linha<1.0e-16 indica espaco altamente distorcido ou elemento muito pequeno." << std::endl;
            std::cout << "J0Linha:" << std::endl << J0Linha << std::endl;
        }

        // Calculo das componentes do versor tangente (unitario)
        vx=Vx/J0Linha; // acima ja verifica J0 proximo de 0
        vy = Vy/J0Linha;
        // std::cout << "Vx:" << std::endl << Vx << std::endl;std::cout << "Vy:" << std::endl << Vy << std::endl;
        // std::cout << "vx:" << std::endl << vx << std::endl;std::cout << "vy:" << std::endl << vy << std::endl;
        // return 0;


     // Calculo da contribuicao forca distribuida em linha
     for (int a = 0; a < nnoselemContorno; ++a) {
        for (int i = 0; i < ndir; ++i) {
            for (int y = 0; y < nnoselemContorno; ++y) {
                int noa =propelemContorno(el,a);
                int noy =propelemContorno(el,y);
                
                double phigama=phi(y); // Calculo phigama
                double phialpha=phi(a); // Calculo phialpha
                
                // Calculo do valor da carga de superficie nodal; a eh o noelem do carregamento
                double Wia=0.0;
                for (int idcar=0; idcar<nnoscar;idcar++){
                 int noidcar=idcarganoSolido(idcar); // no realmente carregado
                 if (noa==noidcar){
                   Wia=carganoSolido(idcar,6+i); // realmente eh idcar, pois carganoSolido tem tamanho nnoscar
                 }
                }

                // Integracao e Montagem vetor de forca externa
                int pos=ndir*(noy-1)+i;
                //Fjext(pos)+= phigama*phialpha*wigauss*J0Linha*Wia;
                PetscScalar valor = phigama * phialpha * wigauss * J0Linha * Wia;
                VecSetValue(Fjext, pos, valor, ADD_VALUES);

            } // Final for noelem y
        } // final for dir i
     } // Final for noelem a


        // Calculo da contribuicao forca distribuida em linha (em termos da pressao e tensao cisalhante p,t1)
    for (int a = 0; a < nnoselemContorno; ++a) {
        int noa =propelemContorno(el,a);
        double phialpha=phi(a); // Calculo phialpha
        // Calculo do valor da carga de superficie nodal; a eh o noelem do carregamento
                 double pressao=0.0, tensaoCis1, tensaoCis2;
                 for (int idcar=0; idcar<nnoscar;idcar++){
                   int noidcar=idcarganoSolido(idcar); // no realmente carregado
                   if (noa==noidcar){
                   pressao=carganoSolido(idcar,0); // pressao
                   tensaoCis1=carganoSolido(idcar,1); // tensao cisalhante 1
                   tensaoCis2=carganoSolido(idcar,2);
                 }
                }

        for (int i = 0; i < ndir; ++i) {
                // Calculo do valor da carga de superficie nodal; a eh o noelem do carregamento
                 double Wia=0.0;
                 if (i==0){
                    Wia=-pressao*vy+tensaoCis1*vx;
                 } else if (i==1){
                    Wia=pressao*vx + tensaoCis1*vy;
                 } else {
                    Wia=0.0;
                    std::cout << "Falta implementar a segunda componente de tensao cisalhante no espaco." << std::endl;
                    return 99;
                 }

            for (int y = 0; y < nnoselemContorno; ++y) {
                
                int noy =propelemContorno(el,y);
                double phigama=phi(y); // Calculo phigama

                // Integracao e Montagem vetor de forca externa
                int pos=ndir*(noy-1)+i;
                //Fjext(pos)+= phigama*phialpha*wigauss*J0Linha*Wia;
                PetscScalar valor = phigama * phialpha * wigauss * J0Linha * Wia;
                VecSetValue(Fjext, pos, valor, ADD_VALUES);
            } // Final for noelem y
             //std::cout << "Wia:" << std::endl << Wia << std::endl; 
        } // final for dir i
        //return 0;
     } // Final for noelem a


     } // Final for igauss
     } // final if no carregado
    } // Final for elementos contorno linha

    // Finaliza a montagem do vetor
     VecAssemblyBegin(Fjext);
     VecAssemblyEnd(Fjext);

  
    //Eigen::VectorXd DUjrecal = (1.0 / npassosCarga) * Ujrecal;
    //Yjf = Yjf + Ujrecal;
    PetscScalar Ftotal, FtotalX=0.0,FtotalY=0.0;
    VecSum(Fjext, &Ftotal);
    for (int noF=0; noF<nnos; ++noF){
        int posX = noF*ndir; int posY = noF*ndir+1;
        PetscScalar valueX, valueY;
        VecGetValues(Fjext, 1, &posX, &valueX); VecGetValues(Fjext, 1, &posY, &valueY);
         FtotalX+= valueX; FtotalY+= valueY;
    }
    

    // Aplicacao incremental
         if (passotempo<=npassosCarga){
         // Dividir carga aplicada
         double passotempoDouble = static_cast<double>(passotempo);
         PetscScalar scale = passotempoDouble/ npassosCarga;
         //Fjext=(Fjext)*(passotempoDouble/npassosCarga); // Fjext=(Fjext)*(passotempoDouble/npassostempo);
         VecScale(Fjext, scale);
         // Udeslo=Udeslo*(passotempo/npassosCarga);
         Ftotal *= scale; //Ftotal=Ftotal*(passotempoDouble/npassosCarga);
         } else {
            // Aplicar carga constante no restante dos passos de tempo
            //Fjext=(Fjext)*(1.0); // o calculo ja eh do valor total, entao basta nao reduzir
         }
   if (rank==0){
    std::cout << "passotempo: " << std::endl << passotempo << std::endl;
    std::cout << "Ftotal:" << std::endl << Ftotal << std::endl; 
    std::cout << "FtotalX:" << std::endl << FtotalX << std::endl;  std::cout << "FtotalY:" << std::endl << FtotalY << std::endl; 
    std::cout << "Ftotal_Atual:" << std::endl << Ftotal << std::endl;
    // return 0;
    std::cout << "status4 - Calculo das forcas nodais: ok\n";
    // return 0;
   }

   //return 0;
    
  //////////////////////FORCAS EXTERNAS ////////////////////////////////////////////////////////////////////////////////////////////
//return 0;
////////////////////// PARTE DO SOLIDO ACIMA ///////////////////////////////////////////////////////////////////////////////////


/*
// imprimir aerofolio IFE
// for nos principais aerofolio - no1 e no2
std::vector<int> nosJaImpressos;

       for (int i=0; i < numBoundElems; i++){
           if (boundary_[i] -> getConstrain(0) == 3 || boundary_[i] -> getConstrain(1) == 3 ){
            // 3: FSinterface ou MOVING
            
               Boundaries::BoundConnect connectB;
               connectB = boundary_[i] -> getBoundaryConnectivity(); // obter vetor de conectividade do elemento i do contorno do fluido

               // Imprimir os valores de connectB
        

            typename Node::VecLocD x;
               for (int m=0; m<2; ++m){
                int no_fluido = connectB(m); // no local m do elemento i de linha do contorno; no global no_fluido
                // Verificar se o nó já foi impresso
               if (std::find(nosJaImpressos.begin(), nosJaImpressos.end(), no_fluido) == nosJaImpressos.end()) {

                x = nodes_[no_fluido]->getCoordinates(); // coordenadas do no no_fluido que fica no contorno do fluido
                double x1_fluido = x(0); double x2_fluido = x(1);
                
                std::cout << "el" << i << ": "; std::cout << no_fluido << " ";
                std::cout << std::setprecision(8) << x1_fluido << " " << x2_fluido << " ";
                std::cout << std::endl;

                // Adicionar nó ao vetor de nós já impressos
                nosJaImpressos.push_back(no_fluido);
               }
               }
           }
       }
 //return 0;
 //*/

// Montar Matriz de conectividade IFE_fazer rastreamento dos pontos do contorno de IFE
        int numBoundNodes=0;
        double toleranciaConectividade = 5.0e-3;
        double tolDx=1.0e-3, tolDy=5.0e-4;
        std::vector<std::vector<int>> tempconectIFE;

        // for nos elementos de contorno
       for (int i=0; i < numBoundElems; i++){
           if (boundary_[i] -> getConstrain(0) == 3 || boundary_[i] -> getConstrain(1) == 3 ){
            // 3: FSinterface ou MOVING
            
               Boundaries::BoundConnect connectB;
               connectB = boundary_[i] -> getBoundaryConnectivity(); // obter vetor de conectividade do elemento i do contorno do fluido

               // passar por todos nos do contorno do fluido
               typename Node::VecLocD x;
               for (int m=0; m<3; ++m){
                int no_fluido = connectB(m); // no local m do elemento i de linha do contorno; no global no_fluido
                x = nodes_[no_fluido]->getCoordinates(); // coordenadas do no no_fluido que fica no contorno do fluido
                double x1_fluido = x(0); double x2_fluido = x(1);

                // passar por todos nos do contorno do solido
                for (int el_j=0; el_j<nelemsContorno; el_j++){
                    for (int n=0; n<3; n++){
                        int no_solido=propelemContorno(el_j,n); // no local n do elemento el_j de linha do contorno; node global solido:  no_solido
                        double x1_solido=coordno(no_solido-1,0);  double x2_solido=coordno(no_solido-1,1); 

                        double distancia_noFLu_noSol = sqrt(pow(x1_fluido - x1_solido, 2.) + pow(x2_fluido - x2_solido, 2.));
                        double dx1=sqrt(pow(x1_fluido - x1_solido,2));
                        double  dx2=sqrt(pow(x2_fluido - x2_solido,2));

                        if ( distancia_noFLu_noSol < toleranciaConectividade && dx1<tolDx && dx2< tolDy ) {
                            numBoundNodes++; //quantificar numero nos no contorno
                            tempconectIFE.push_back({no_solido, no_fluido}); // guardando nodes da conectividade IFE
                            //std::cout << "distancia_noFLu_noSol:" << std::endl << distancia_noFLu_noSol << std::endl; 
                            //continue;
                        }
                    }
                }
                }
                     // std::cout << "Coord X atual " << nodes_[no2]->getUpdatedCoordinates()(0) << std::endl;
                     // std::cout << "\n " << std::endl;
           };
       };
       //return 0;
            

    // Remover linhas duplicadas
     std::sort(tempconectIFE.begin(), tempconectIFE.end());
     auto last = std::unique(tempconectIFE.begin(), tempconectIFE.end());
     tempconectIFE.erase(last, tempconectIFE.end());

    // atualizando numero de elementos no contorno
     numBoundNodes = tempconectIFE.size();

    // Verificar se todos os elementos são diferentes
     if (numBoundNodes != std::distance(tempconectIFE.begin(), last)) {
         std::cout << "Aviso: Elementos repetidos encontrados após a remoção das duplicatas na matriz de conectividade." << std::endl;
         return 99;
     }

    // Verificar o numero de nos no contorno: nnosContorno = nnoselem
    nnosContorno=3+(nelemsContorno-2)*2+1;
    if (nnosContorno!=numBoundNodes){
        std::cout << "Aviso: captura dos nos do contorno esta errada ou a especificacao dos elementos do contorno." << std::endl;
         return 99;
    }
    
    // Transferir dados matriz Eigen
    MatrixXi conectIFE = MatrixXi::Zero(numBoundNodes, 2);
    for (int i = 0; i < numBoundNodes; ++i) {
        conectIFE(i, 0) = tempconectIFE[i][0]; // Montagem matriz de conectividade IFE
        conectIFE(i, 1) = tempconectIFE[i][1];
    }
    if (rank==0){
        std::cout << "numBoundNodes: " << std::endl << numBoundNodes << std::endl;
    }

/*
    if (rank==0){
    
    std::cout << "conectIFE:" << std::endl << conectIFE << std::endl; 
    std::cout << "numBoundNodes: " << std::endl << numBoundNodes << std::endl;

     int no_solido=392, no_fluido=81-1;

    typename Node::VecLocD x;    
    x = nodes_[no_fluido]->getCoordinates(); // coordenadas do no no_fluido que fica no contorno do fluido
    double x1_fluido = x(0); double x2_fluido = x(1);

    double x1_solido=coordno(no_solido-1,0);  double x2_solido=coordno(no_solido-1,1); 

    double distancia_5 = sqrt(pow(x1_fluido - x1_solido, 2.) + pow(x2_fluido - x2_solido, 2.));

    std::cout << "x1_fluido:" << std::endl << x1_fluido << std::endl; std::cout << "x2_fluido:" << std::endl << x2_fluido << std::endl; 
    std::cout << "x1_solido:" << std::endl << x1_solido << std::endl; std::cout << "x2_solido:" << std::endl << x2_solido << std::endl; 
    std::cout << "distancia_5:" << std::endl << distancia_5 << std::endl; 


     x1_solido=coordno(388,0);   x2_solido=coordno(388,1); 
    std::cout << "x1_solido:" << std::endl << x1_solido << std::endl; std::cout << "x2_solido:" << std::endl << x2_solido << std::endl; 

     x1_solido=coordno(389,0);   x2_solido=coordno(389,1); 
    std::cout << "x1_solido:" << std::endl << x1_solido << std::endl; std::cout << "x2_solido:" << std::endl << x2_solido << std::endl; 
   }
   */


// -> atualizar no fluido: velocidade inicial do solido (Dirichlet), (usar Matriz de conectividade e interpolação)


double DYkmax = 1000.0;
int contDYkmax = 0;
         
        for (int inewton = 0; inewton < iterNumber2; inewton++){

            // Acoplamento //////////////////////////////////////////////////////////////////////////
             bool atualizarVelocidade = false;
             bool atualizarPosicaoMalha = false;
             bool atualizarForcasSuperficie = true;

            if (atualizarVelocidade){

            // atualizar velocidade no fluido com Velocidade_Solido (Direchlet no fluido)
            for (int no=0; no<numBoundNodes; ++no){
                int noSolido = conectIFE(no,0);
                int noFluido = conectIFE(no,1);

                // pegar velocidade no solido
                double velo_x = Vjf(noSolido*2);
                double velo_y = Vjf(noSolido*2+1);

                // aplicar velocidades no fluido
                // buscar elemento relativo ao noFluido
                // verificar rank MPI
               // if (rank_elem==rank){
                    //Atualizacao das velocidades
                    typename Node::VecLocD v;
                      v = nodes_[noFluido]->getVelocity();
                      v(0) = velo_x; // substituicao 
                      v(1) = velo_y;
                     
                      nodes_[noFluido] -> setVelocity(v);  //nodes_[noFluido] -> setUpdatedVelocity(v);
                      // nodes_[i] -> incrementVelocity(1,val); 
                //}
            }
            }





            boost::posix_time::ptime t1 =                             
                               boost::posix_time::microsec_clock::local_time();
            
            ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                                2*numNodes+numNodes, 2*numNodes+numNodes,
                                100,NULL,300,NULL,&A); 
            CHKERRQ(ierr);
            
            ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);
            
            //Create PETSc vectors
            ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
            ierr = VecSetSizes(b,PETSC_DECIDE,2*numNodes+numNodes);
            CHKERRQ(ierr);
            ierr = VecSetFromOptions(b);CHKERRQ(ierr);
            ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
            ierr = VecDuplicate(b,&All);CHKERRQ(ierr);
            
            //std::cout << "Istart = " << Istart << " Iend = " << Iend << std::endl;

            for (int jel = 0; jel < numElem; jel++){   
                
                //if (part_elem[jel] == rank) {
                    //Compute Element matrix
                    elements_[jel] -> getTransientNavierStokes();

                    typename Elements::LocalMatrix Ajac;
                    typename Elements::LocalVector Rhs;
                    typename Elements::Connectivity connec;
                    
                    //Gets element connectivity, jacobian and rhs 
                    connec = elements_[jel] -> getConnectivity();
                    Ajac = elements_[jel] -> getJacNRMatrix();
                    Rhs = elements_[jel] -> getRhsVector();
                    
                    //Disperse local contributions into the global matrix
                    //Matrix K and C
                    for (int i=0; i<6; i++){
                        for (int j=0; j<6; j++){
                            if (fabs(Ajac(2*i  ,2*j  )) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * connec(j);
                                ierr = MatSetValues(A, 1, &dof_i,1, &dof_j,
                                                    &Ajac(2*i  ,2*j  ),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i+1,2*j  )) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i+1,2*j  ),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i  ,2*j+1)) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * connec(j) + 1;
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i  ,2*j+1),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i+1,2*j+1)) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * connec(j) + 1;
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i+1,2*j+1),
                                                    ADD_VALUES);
                            };
                        
                            //Matrix Q and Qt
                            if (fabs(Ajac(2*i  ,12+j)) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i  ,12+j),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(12+j,2*i  )) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_j, 1, &dof_i,
                                                    &Ajac(12+j,2*i  ),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i+1,12+j)) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i+1,12+j),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(12+j,2*i+1)) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_j, 1, &dof_i,
                                                    &Ajac(12+j,2*i+1),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(12+i,12+j)) >= 1.e-15){
                                int dof_i = 2 * numNodes + connec(i);
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(12+i,12+j),
                                                    ADD_VALUES);
                            };
                        };
                        
                        //Rhs vector
                        if (fabs(Rhs(2*i  )) >= 1.e-15){
                            int dof_i = 2 * connec(i);
                            ierr = VecSetValues(b, 1, &dof_i, &Rhs(2*i  ),
                                                ADD_VALUES);
                        };
                        
                        if (fabs(Rhs(2*i+1)) >= 1.e-15){
                            int dof_i = 2 * connec(i)+1;
                            ierr = VecSetValues(b, 1, &dof_i, &Rhs(2*i+1),
                                                ADD_VALUES);
                        };
                        if (fabs(Rhs(12+i)) >= 1.e-15){
                            int dof_i = 2 * numNodes + connec(i);
                            ierr = VecSetValues(b, 1, &dof_i, &Rhs(12+i),
                                                ADD_VALUES);
                        };
                    };
                //};
            }; //Elements
            
            //Assemble matrices and vectors
            ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            
            ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
            ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
            

            // Mat Aperm;
            // MatGetOrdering(A,MATORDERINGRCM,&rowperm,&colperm);
            // MatPermute(A,rowperm,colperm,&Aperm);
            // VecPermute(b,colperm,PETSC_FALSE);
            // MatDestroy(&A);
            // A    = Aperm;    

            //MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            //MatView(A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
            //ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            
            //Create KSP context to solve the linear system
            ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
            
            ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
            
            // ierr = KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,
            //                         500);CHKERRQ(ierr);
            
            // ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
            
            // // ierr = KSPGetPC(ksp,&pc);
            
            // // ierr = PCSetType(pc,PCNONE);
            
            // // ierr = KSPSetType(ksp,KSPDGMRES); CHKERRQ(ierr);

            // ierr = KSPGMRESSetRestart(ksp, 500); CHKERRQ(ierr);
            
            // //    ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);
            

        // //   //   ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nullsp);
        // // // ierr = MatSetNullSpace(A, nullsp);
        // // // ierr = MatNullSpaceDestroy(&nullsp);

   

#if defined(PETSC_HAVE_MUMPS)
            ierr = KSPSetType(ksp,KSPPREONLY);
            ierr = KSPGetPC(ksp,&pc);
            ierr = PCSetType(pc, PCLU);
#endif          
            ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
            ierr = KSPSetUp(ksp);



            ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

            ierr = KSPGetTotalIterations(ksp, &iterations);            

            // VecPermute(u,rowperm,PETSC_TRUE);

            //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);CHKERRQ(ierr);
            
            //Gathers the solution vector to the master process
            ierr = VecScatterCreateToAll(u, &ctx, &All);CHKERRQ(ierr);
            
            ierr = VecScatterBegin(ctx, u, All, INSERT_VALUES, SCATTER_FORWARD);
            CHKERRQ(ierr);
            
            ierr = VecScatterEnd(ctx, u, All, INSERT_VALUES, SCATTER_FORWARD);
            CHKERRQ(ierr);
            
            ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

            
            //Updates nodal values
            double p_;
            duNorm = 0.;
            double dpNorm = 0.;
            Ione = 1;
            
            for (int i = 0; i < numNodes; ++i){
                //if (nodes_[i] -> getConstrains(0) == 0){
                    Ii = 2*i; // posicao da velocidade na direcao x1
                    ierr = VecGetValues(All, Ione, &Ii, &val);CHKERRQ(ierr); // obtem valor nodal de Vx
                    duNorm += val*val;
                    nodes_[i] -> incrementVelocity(0,val); // atualiza valor nodal Vx
                    //}; 
                
                    //if (nodes_[i] -> getConstrains(1) == 0){
                    Ii = 2*i+1; // posicao da velcidade na direcao x2
                    ierr = VecGetValues(All, Ione, &Ii, &val);CHKERRQ(ierr); // obtem valor nodal Vy
                    duNorm += val*val;
                    nodes_[i] -> incrementVelocity(1,val); // att valor nodal Vy
                    //};
            };
            
            for (int i = 0; i<numNodes; i++){
                Ii = 2*numNodes+i; // posicao pressao; pressao esta na parte final do vetor All; pois comecao em numNodes*2
                ierr = VecGetValues(All,Ione,&Ii,&val);CHKERRQ(ierr); // obtem valor nodal da pressao
                p_ = val;
                dpNorm += val*val;
                nodes_[i] -> incrementPressure(p_); // att valor nodal p
            };


/*
            // como acessar valores nodais com get
              for (int i=0; i<numNodes; i++){
                    
                    // obter velocidades no fluido
                    double Vx1 = nodes_[i] -> getVelocity(0);            
                    double Vx2 = nodes_[i] -> getVelocity(1);
                    
                    // Obter pressao
                    double pressao = nodes_[i] -> getPressure(); 
                    
                    // obter Vorticidade
                    double vorticidade = nodes_[i] -> getVorticity(); 
                    
                    // obter velocidade da malha
                    double Vx1Malha = nodes_[i] -> getMeshVelocity(0); 
                    double Vx2Malha = nodes_[i] -> getMeshVelocity(1);
                    
                    // obter coordenadas
                    //double coorno_noi_atual = nodes_[i]->getCoordinates();
                   // double coorno_noi_inicial = nodes_[i]->getInitialCoordinates();

                    // Obter coordenadas
                     typename Node::VecLocD coorno_noi_atual = nodes_[i]->getCoordinates();
                     typename Node::VecLocD coorno_noi_inicial = nodes_[i]->getInitialCoordinates();
                    
                    // obter restricoes
                    int restricao1 = nodes_[i]->getConstrains(0);
                    int restricao2 = nodes_[i]->getConstrains(1);

                   //  std::cout << "Vx1=" << Vx1 << std::endl; std::cout << "Vx2=" << Vx2 << std::endl; std::cout << "pressao=" << pressao << std::endl;
                   //  std::cout << "vorticidade=" << vorticidade << std::endl; 
                   //  std::cout << "Vx1Malha=" << Vx1Malha << std::endl; std::cout << "Vx2Malha=" << Vx2Malha << std::endl;
                   //  std::cout << "coorno_noi_atual=(" << coorno_noi_atual(0) << ", " << coorno_noi_atual(1) << ")" << std::endl;
                   //  std::cout << "restricao1=" << restricao1 << std::endl; std::cout << "restricao2=" << restricao2 << std::endl;
                }; 
                //return 0;
                */

// calculo tensoes de cauchy
// Calcula gradiente da velocidade: D_Phi.(D_ksi/dY).Velocidade; (Dksi_Dy= A1⁻1)
// calculo do desviador
// calculo hidrostatico
// tensor de cauchy: desviador + hidrostatico

//------------------------------------------------------------------------------
//---------------------- CALCULO FORCAS DE SUPERFICIE E ACOPLAMENTO-----------------------------
//------------------------------------------------------------------------------
    // reinicializacao variaveis globais
    carganoSolido = MatrixXd::Zero(nnoscar, 3 * 4); //MatrixXd carganoSolido = MatrixXd::Zero(nnoscar, 3 * 4);
      // condicao valida para solidos totalmente imersos
      // solidos parcialmente imersos sera necessario buscar area molhada ou permitir descontinuidades valores nodais

    // inicializacao das variaveis
    //double pressureDragCoefficient = 0.;
    
    // Loop principal sobre os elementos de contorno do fluido
    for (int jel = 0; jel < numBoundElems; jel++){
           // verificar se o elemento de contorno pertence a interface fluido-estrutura   
           if (boundary_[jel] -> getConstrain(0) == 3 || boundary_[jel] -> getConstrain(1) == 3 ){
            // 3: FSinterface ou MOVING

        // reinicializacao das variaveis locais
        double fx_no1 = 0.;
        double fy_no1 = 0.;
        double fx_no2 = 0.;
        double fy_no2 = 0.;
        double fx_no3 = 0.;
        double fy_no3 = 0.;
        
                int iel = boundary_[jel] -> getElement(); // obtem indice do elemento 2D iel associado ao elemento 1D jel
                
                // itera sobre todos elementos para pegar o correspondente a iel
                for (int j = 0; j < numElem; ++j){
                    if (elements_[j] -> getIndex() == iel){
                        elements_[j] -> computeDragAndLiftForces(); // realiza todos os calculos

                        fx_no1 = elements_[j] -> getForcaSuperficieFxno1(); // recebe forca de superficie na dir x do node local no1
                        fy_no1 = elements_[j] -> getForcaSuperficieFyno1();

                        fx_no2 = elements_[j] -> getForcaSuperficieFxno2();
                        fy_no2 = elements_[j] -> getForcaSuperficieFyno2();

                        fx_no3 = elements_[j] -> getForcaSuperficieFxno3();
                        fy_no3 = elements_[j] -> getForcaSuperficieFyno3();
                    }   
                }

            // Transferencia dos valores nodais de forca de superficie para a Matriz de Forcas Externas do Solido: cargano; idcargano
                Boundaries::BoundConnect connectB;
                connectB = boundary_[jel] -> getBoundaryConnectivity(); // obter vetor de conectividade do elemento jel do contorno do fluido

                // obter cada node do elemento de contorno do fluido
                   typename Node::VecLocD x; // ele usa numeracao sequencial em linha para 1D
                   int no1_fluido = connectB(0); // no local 1 do elemento jel de linha do contorno; no global no_fluido
                   int no2_fluido = connectB(1); // no local 2 do elemento jel de linha do contorno; no global no_fluido
                   int no3_fluido = connectB(2); // no local 3 do elemento jel de linha do contorno; no global no_fluido
                   
                // verificar se o node ja esta presente
                // criar matriz cargano com tamanho global: {nnosx9}
                // criar vetor idcargano com tamanho global: {nnos}
                
                // atualizacao da matriz de carregamento do solido
                // verificar se ha valor, se houver: calcular media e setar; senao: substituir pelo valor
                // fx_no1
                if (carganoSolido(no1_fluido,6+0) != 0){
                    carganoSolido(no1_fluido,6+0) = 0.5*(carganoSolido(no1_fluido,6+0) + fx_no1);
                } else {
                    carganoSolido(no1_fluido,6+0) = 0.5*(carganoSolido(no1_fluido,6+0) + fx_no1);
                }
                // fy_no1
                if (carganoSolido(no1_fluido,6+1) != 0){
                    carganoSolido(no1_fluido,6+1) = 0.5*(carganoSolido(no1_fluido,6+1) + fy_no1);
                } else {
                    carganoSolido(no1_fluido,6+1) = 0.5*(carganoSolido(no1_fluido,6+1) + fy_no1);
                }

                // fx_no2
                if (carganoSolido(no2_fluido,6+0) != 0){
                    carganoSolido(no2_fluido,6+0) = 0.5*(carganoSolido(no2_fluido,6+0) + fx_no2);
                } else {
                    carganoSolido(no2_fluido,6+0) = 0.5*(carganoSolido(no2_fluido,6+0) + fx_no2);
                }
                // fy_no2
                if (carganoSolido(no2_fluido,6+1) != 0){
                    carganoSolido(no2_fluido,6+1) = 0.5*(carganoSolido(no2_fluido,6+1) + fy_no2);
                } else {
                    carganoSolido(no2_fluido,6+1) = 0.5*(carganoSolido(no2_fluido,6+1) + fy_no2);
                }

                // fx_no3
                if (carganoSolido(no3_fluido,6+0) != 0){
                    carganoSolido(no3_fluido,6+0) = 0.5*(carganoSolido(no3_fluido,6+0) + fx_no3);
                } else {
                    carganoSolido(no3_fluido,6+0) = 0.5*(carganoSolido(no3_fluido,6+0) + fx_no3);
                }
                // fy_no3
                if (carganoSolido(no3_fluido,6+1) != 0){
                    carganoSolido(no3_fluido,6+1) = 0.5*(carganoSolido(no3_fluido,6+1) + fy_no3);
                } else {
                    carganoSolido(no3_fluido,6+1) = 0.5*(carganoSolido(no3_fluido,6+1) + fy_no3);
                }
                
                // 
           } // final if
    }; // final for elementos contorno

    // sincroniza os processos MPI
    MPI_Barrier(PETSC_COMM_WORLD);

    // double totalDragCoefficient = 0.;
    // double totalLiftCoefficient = 0.;
    // // MPI_Allreduce combina os valores de cada processo, reune e distrubui o valor final para todos os processos
    // MPI_Allreduce(&dragCoefficient,&totalDragCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD); // soma os valores locais da variavel dragCoefficiente em total
    // MPI_Allreduce(&liftCoefficient,&totalLiftCoefficient,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD); // 1 is the number of elements in the buffer

    // impressao  
    if (rank == 0) {
        const int timeWidth = 15;
        const int numWidth = 15;

        // Impressão dos resultados na janela de comando
        std::cout << "Time: " << std::setw(15) << iTimeStep * dTime;
        std::cout << "carganoSolido: " << carganoSolido << std::endl;
    }

    return 0;

            
            // calculo vetor de forcas nodais equivalentes: realizado dentro do programa solid

// Calculo do solido MNR //////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////LACO WHILE - SISTEMA NAO LINEAR SOLIDO ////////////////////////////////////////////////////////////////////////////////////////////
//LACO WHILE POSICAO EQUILIBRIO DYKMAX - NEWTON RAPHSON
    
//for (int inewton = 0; inewton < iterNumber2; inewton++){
//while (DYkmax >= tol) {
        contDYkmax++;

        if (contDYkmax >= limitecont) {
            std::cout << "Limite contDYkmax DYkmax=" << limitecont << " atingido." << std::endl;
            std::cout << "Verifique o arquivo de entrada de dados ou altere o limitecont." << std::endl;
            return 1;
        }

        // reinicialização da Hessiana Hoz // att posicao já é feita em Y, E, S, l // a cada atualizacao da posicao DYk é preciso zerar Hoz e Fjint
        // Hd = Eigen::SparseMatrix<double>(nglt, nglt);
        // Hoz = Eigen::SparseMatrix<double>(nglt, nglt);
        // Mp = Eigen::SparseMatrix<double>(nglt, nglt);
        // Mm = Eigen::SparseMatrix<double>(nglt, nglt); 
           // Inicializa com zeros
            MatZeroEntries(Hd);
            MatZeroEntries(Hoz);
            MatZeroEntries(Mm);
            MatZeroEntries(Mp);
            MatZeroEntries(Cc); // a forma de montagem PETSc esta Cc=Cc+Mm.k+Hoz.k

            // Finaliza a montagem das matrizes
             MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY);
             MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY);
             
             MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
             MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
             
             MatAssemblyBegin(Mm, MAT_FINAL_ASSEMBLY);
             MatAssemblyEnd(Mm, MAT_FINAL_ASSEMBLY);
             
             MatAssemblyBegin(Mp, MAT_FINAL_ASSEMBLY);
             MatAssemblyEnd(Mp, MAT_FINAL_ASSEMBLY);

             MatAssemblyBegin(Cc, MAT_FINAL_ASSEMBLY);
             MatAssemblyEnd(Cc, MAT_FINAL_ASSEMBLY);
            // recalcula-se massa a cada DYk quando promove-se remocao de elementos por ruptura ou por otimizacao 
     
            //Eigen::VectorXd Fjint = Eigen::VectorXd::Zero(nglt); 
               VecSet(Fjint, 0.0);
            // Finaliza a montagem do vetor
              VecAssemblyBegin(Fjint);
              VecAssemblyEnd(Fjint);
    

//FOR ELEMENTOS - Superficie
//FOR ELEMENTOS - Superficie
for (int el = 0; el < nelems; ++el) {
    // trocar nelems -> nelemSuperficie; Assim temos certeza que o elemento pertence a superficie 2D - Integracao Hammer
        // submatrizes auxiliares
        int idmat = propelem(el, nnoselem);
        Eigen::VectorXd pmat = propmat.row(idmat - 1);

// for pontos de quadratura superficie
for (int ihammer=0; ihammer<nhammer; ++ihammer){
    //tipoelem ja definido //tipoelem: 1 solido 3D; 2 chapa EPT; 3 chapa EPD; 4 placa; 5 casca; 6 portico 3D; 

    //Mapeamento inicial do elemento //A0
      Eigen::MatrixXd Xil = Eigen::MatrixXd::Zero(nnoselemSuperficie,ndirSuperficie); // Vale apenas para superficie 2D
      for (int i = 0; i < ndirSuperficie; ++i) {
      for (int y = 0; y < nnoselemSuperficie; ++y) {
        int noy=propelemSuperficie(el, y);  // Pega da Matriz de conectividade da Superficie
        Xil(y,i)+=coordno(noy-1,i);  // Xil -> Xiy, dir i, noelem y; Matriz coordenadas superficie do elemento 2D
       }
      }
     //std::cout << "Xil=" << std::endl << Xil << std::endl;
     int tipoFuncaoForma=1; //1 para Triangular; 2 para Tetraedrica; 3 para quadrangular; 4 para hexaedrica; 5 para prismatrica;
    
     Eigen::VectorXd ksi = Eigen::VectorXd::Zero(ndirSuperficie);
     Eigen::VectorXd phi = Eigen::VectorXd::Zero(nnoselemSuperficie);
     Eigen::MatrixXd dphi_dksi = Eigen::MatrixXd::Zero(nnoselemSuperficie,ndirSuperficie); 
     Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
     Eigen::MatrixXd inversaA0 = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);

     double ksi1 = Matrizksihammer(ihammer,0);
     double ksi2 = Matrizksihammer(ihammer,1);
     double ksi3 = 0.0;
    
     ksi(0)=ksi1;
     ksi(1)=ksi2;
     if (tipoFuncaoForma==2 ||  tipoFuncaoForma==4 || tipoFuncaoForma==5){
      ksi(2)=ksi3; //se tiver 3 direcoes; ou seja solido 3D
     }

     //FuncaoFormaTriangulo(tipoFuncaoForma, ndirSuperficie,  grau,  ksi, phi, dphi_dksi);
     FuncaoFormaTriangulo(coef, dcoef_dksi1, dcoef_dksi2, grau, ksi,  phi,  dphi_dksi); // a funcao retorna phi e dphi_dksi calculados
 
   // Propriedades Geometricas Iniciais
    
     // Matriz Gradiente da Funcao Mudanca de Configuracao Inicial A0
     //A0=dphi_dksi.transpose()*Xil;
        A0=Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
          for (int i=0;i<ndirSuperficie;++i){
          for (int j=0;j<ndirSuperficie;++j){
            for (int no=0;no<nnoselemSuperficie;++no){
                A0(i,j)+=dphi_dksi(no,j)*Xil(no,i);

             }
          }
          }
     //std::cout << "A0:\n" << A0 << std::endl;
    
     //Jacobiano de A0 - Determinante da Matriz Mudanca de Configuracao Inicial
     double J0 = A0.determinant();    

         if (J0 < 10e-8) {
         std::cout << "Mapeamento Inicial Problematica, pois J0 do elemento de Superficie: " << el << " nulo ou menor que 10e-8." << std::endl;
         std::cout << "Verifique mapeamento inicial hessiana e forcas internas." << std::endl;
         return 1;
        }

    
    //Mapeamento Atual do elemento //A1
      Eigen::MatrixXd Yil = Eigen::MatrixXd::Zero(nnoselemSuperficie,ndirSuperficie); // Vale apenas para superficie 2D
      for (int i = 0; i < ndirSuperficie; ++i) {
      for (int y = 0; y < nnoselemSuperficie; ++y) {
        int noy=propelemSuperficie(el, y); //noy = noglobal do no local y
        int posy=ndirSuperficie*(noy-1)+i;
        //Yil(y,i)+=Yjf(posy);  // Yil -> Yiy, dir i, noelem y; Matriz coordenadas superficie do elemento 2D
        PetscScalar value;
        VecGetValues(Yjf, 1, &posy, &value);
         Yil(y,i) += value;
       }
      }
   
//    if (rank==0){
//     std::cout << "****************** status Atual: ok *********************\n";
// }
    Eigen::MatrixXd A1 = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    //dphi_dksi - Gradiente da funcao Phi ja calculado no ponto de hammer
    //std::cout << "Yjf=" << Yjf  << std::endl;

    // Matriz Gradiente da Funcao Mudanca de Configuracao Atual A1
    //A1=dphi_dksi.transpose()*Yil;
    A1=Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    for (int i=0;i<ndirSuperficie;++i){
        for (int j=0;j<ndirSuperficie;++j){
            for (int no=0;no<nnoselemSuperficie;++no){
                A1(i,j)+=dphi_dksi(no,j)*Yil(no,i);

            }
        }
    }


    //std::cout << "A1:\n" << A1 << std::endl; std::cout << "Yil=" << Yil  << std::endl; std::cout << "A1=" << A1  << std::endl;

    //Jacobiano de A1 - Determinante da Matriz Mudanca de Configuracao Atual
     double J1 = A1.determinant();

      if (J1 < 10e-8) {
            std::cout << "Mapeamento Atual Problematico, pois J1 do elemento: " << el << " nulo ou menor que 10e-8." << std::endl;
            std::cout << "Verifique o arquivo de entrada de dados: dentro do mapeamento atual A1 hessiana e forcas internas." << std::endl;
            return 1;
        }
     

    //Peso de hammer
    wihammer=Vetorwih(ihammer);

    ////////////////////////////////////////
    // Propriedades de Deformacao Geometrica E, DE
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie); //tensor de along cauchy a direita - difere do Cc Matriz de amortecimento
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ndirSuperficie, ndirSuperficie);
    Eigen::MatrixXd Enl = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd Snl = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    
    A=A1*A0.inverse(); //gradiente da funcao mudanca de configuracao
    //A=A0.inverse()*A1; //gradiente da funcao mudanca de configuracao
    //std::cout << "A total:\n" << A << std::endl;
    
    C=A.transpose()*A; //tensor de alongamento a direita de cauch green
    //std::cout << "C:\n" << C << std::endl; std::cout << "I:\n" << I << std::endl;

    Enl = 0.5*(C-I); // tensor de deformacoes de green lagrange
    //std::cout << "Enl:\n" << Enl << std::endl;

    //double tol=1.0e-20;
    //for (int i=0; i <ndirSuperficie; ++i){
    //    for (int j=0; j <ndirSuperficie; ++j){
    //        if (std::abs(Enl(i,j)) < tol) {
     //          Enl(i,j) = 0.0; // Define valores muito proximos de zero como zero
    //           }
    //    }
    //}

    double Ev=0; //Traco do tensor de deformacoes ou deformacao volumetrica
    for (int i=0; i <ndirSuperficie; ++i){
        for (int j=0; j <ndirSuperficie; ++j){
       if (i==j){
         Ev+=Enl(i,j);
       }
     } 
    } 
    
    //Incrementar energia de deformacao interna
    //ue=E11+...
    //Uint+=ue*wh(ihammer)*J0;  
    
    //Calculo propriedades dos Elementos
    double K,vmat,ro, aT, Gmat; //tipoelem: 1 solido 3D; 2 chapa EPT; 3 chapa EPD; 4 placa; 5 casca; 6 portico 3D; 
    
    if (tipoelem == 1) {
       K =propmat(idmat-1,0); vmat =propmat(idmat-1,1); ro =propmat(idmat-1,2); aT =propmat(idmat-1,3);

    } else if (tipoelem == 2) {
       K =propmat(idmat-1,0); vmat =propmat(idmat-1,1); ro =propmat(idmat-1,2); aT =propmat(idmat-1,3);
       
       
    } else if (tipoelem == 3) {
       K =propmat(idmat-1,0); vmat =propmat(idmat-1,1); ro =propmat(idmat-1,2); aT =propmat(idmat-1,3);

  
    } else {
    std::cout << "Elemento " << tipoelem;
    std::cout << " ainda nao implementada." << std::endl;
    return 1;
    }
    Gmat = K/(2.0*(1.0+vmat)); 

    // Cálculo do comprimento característico Lokmin, Lokmax
     double Lo=0.0; // Implementar calculo maior lado e menor lado 
      if (Lo < Lokmin) {
       Lokmin = Lo;
       elLokmin = el;
       Kmin = K;
       romin = ro;
      }

     if (Lo > Lokmax) {
      Lokmax = Lo;
      elLokmax = el;
      Kmax = K;
      romax = ro;
     }

    // Calculo Deformacoes e Tensoes
    // K é igual ao E do módulo de Yong ou rigidez do material longitudinal
    // E passa a ser a deformação não linear de Green Lagrange


    // tipotensao_1(SaintVenantKirchhoff) || 2(Hooke) || 3(AlmansisLinear)
     double Emat=K; 
     double lambdaMat=Emat*vmat/((1.0-2.0*vmat)*(1.0+vmat));
     double kmat=Emat/((1.0+vmat)*(1.0-2.0*vmat));
     int deltaij=0;
    
    if (tipotensao == 1) {
     // Calculo tensor de tensoes SVK Snl ndirxndir

    //tipoelem: 1 solido 3D; 2 chapa EPT; 3 chapa EPD; 4 placa; 5 casca; 6 portico 3D; 
    if (tipoelem==1){

     //Solido 3D
    for (int i=0; i <ndir; ++i){
      for (int j=0; j <ndir; ++j){
       if (i==j){
         deltaij=1;
       }
       Snl(i,j)=2*Gmat*Enl(i,j)+lambdaMat*Ev*deltaij;
     } 
    } 

    }
    else if(tipoelem==2){
     //Chapa EPT
    Snl(0,0)=(2*Gmat/(1-vmat))*(Enl(0,0)+vmat*Enl(1,1));
    Snl(0,1)= 2*Gmat*Enl(0,1);
    Snl(1,0)=2*Gmat*Enl(1,0);
    Snl(1,1)=(2*Gmat/(1-vmat))*(Enl(1,1)+vmat*Enl(0,0));

    }
    else if(tipoelem==3){
     //Chapa EPD
    Snl(0,0)=kmat*((1-vmat)*Enl(0,0)+vmat*Enl(1,1));
    Snl(0,1)= 2*Gmat*Enl(0,1);
    Snl(1,0)=2*Gmat*Enl(1,0);
    Snl(1,1)=kmat*((1-vmat)*Enl(1,1)+vmat*Enl(0,0));

    }
    else if(tipoelem==4){
        //4 placa; 

    }
    else if(tipoelem==5){
       //5 casca; 
    }
    else if(tipoelem==6){
      //6 portico 3D; 
    }
    else {
        std::cout << "Tipo de elemento nao implementado." << std::endl;
        //return 50; //tipo de elemento nao implementado
    }
    //std::cout << "el:\n" << el << std::endl; std::cout << "ihammer:\n" << ihammer << std::endl;  std::cout << "kmat:\n" << kmat << std::endl;
    //std::cout << "Gmat:\n" << Gmat << std::endl; std::cout << "kmat:\n" << kmat << std::endl; std::cout << "kmat:\n" << kmat << std::endl;
    //std::cout << "Snl:\n" << Snl << std::endl;

    } else if (tipotensao == 2) {
         // tipotensao_1(SaintVenantKirchhoff) || 2(Hooke) || 3(AlmansisLinear)
      

    } else if (tipotensao == 3) {
        // tipotensao_1(SaintVenantKirchhoff) || 2(Hooke) || 3(AlmansisLinear)

    } else {
      std::cout << "Tipotensao " << tipotensao;
      std::cout << " ainda nao implementada." << std::endl;
      return 1;
    }
    

    // Integracao e Montagem do vetor de forças nodais internas
    // Reinicialização de Fjint é apenas a cada passo de DYk e não a cada el
    // Contribuição do elemento el
     
    ////////////////////////////////////////// Propriedades geometricas para a Hessiana e forca interna
    //Tensor derivada do gradiente mudanca de configuracao: DA1ai
    Eigen::MatrixXd DEai = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd D2Eaibj = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    //Eigen::MatrixXd DSaj = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd DSai = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd DA1ai = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd DA1bj = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);
    Eigen::MatrixXd DEbj = Eigen::MatrixXd::Zero(ndirSuperficie,ndirSuperficie);

     rows.clear();
     cols.clear();
     values.clear();

     for (int noelema=0; noelema<nnoselemSuperficie; ++noelema){
     for (int diri = 0; diri < ndirSuperficie; ++diri) {

        //Calculo DA1ai
        for (int i=0; i <ndirSuperficie; ++i){
           for (int j=0; j <ndirSuperficie; ++j){
              int deltakronecker=0;
              if (diri==i) {
                  deltakronecker=1;
              }
              DA1ai(i,j)=dphi_dksi(noelema,j)*deltakronecker;
            } 
        }
         //std::cout << "noelema:\n" << noelema << std::endl; std::cout << "diri:\n" << diri << std::endl; std::cout << "DA1ai:\n" << DA1ai << std::endl;
     
        //Calculo DEai
        inversaA0=A0.inverse();
        DEai=0.5*(inversaA0.transpose()*DA1ai.transpose()*A1*inversaA0+inversaA0.transpose()*A1.transpose()*DA1ai*inversaA0);
        //std::cout << "DEai:\n" << DEai << std::endl;

        // Calculo fia
         double fia=0.0;
         for (int dir1=0; dir1<ndirSuperficie; ++dir1){
         for (int dir2 = 0; dir2 < ndirSuperficie; ++dir2) {
            fia+=DEai(dir1,dir2)*Snl(dir1,dir2); // contracao dupla fia= DEai : Snl
         }
         }


        // Integracao e montagem vetor global Fjint
         int noa=propelemSuperficie(el,noelema);
         int posFjint=ndirSuperficie*(noa-1)+diri;
        // Fjint(posFjint)+=fia*wihammer*J0; // integrando e montando vetor global;
         PetscScalar valor = fia * wihammer * J0;
         VecSetValue(Fjint, posFjint, valor, ADD_VALUES);

         
    // Hessiana
     //Lacos Globais da hessiana
     for (int noelemb=0; noelemb <nnoselem; ++noelemb){
     for (int dirj=0; dirj <ndirSuperficie; ++dirj){

       //Calculo DA1bj
        for (int i=0; i <ndirSuperficie; ++i){
           for (int j=0; j <ndirSuperficie; ++j){
              int deltakronecker=0;
              if (dirj==i) {
                  // conferir se eh dirj==j ou  dirj==i
                  deltakronecker=1;
              }
              DA1bj(i,j)=dphi_dksi(noelemb,j)*deltakronecker;
            } 
        }
        
        // Calculo DA1ai, DEai prontos

        //Calculo DEbj
        DEbj=0.5*(inversaA0.transpose()*DA1bj.transpose()*A1*inversaA0+inversaA0.transpose()*A1.transpose()*DA1bj*inversaA0);
       
        //Traco DEai
         double DEv=0; //Traco do tensor de deformacoes ou deformacao volumetrica
         for (int i=0; i <ndirSuperficie; ++i){
          for (int j=0; j <ndirSuperficie; ++j){
           if (i==j){
            DEv+=DEai(i,j); // DEai, pois DSai
           }
         } 
        }

       //Calculo DSai
        if (tipoelem==1){
         //Solido 3D
          for (int i=0; i <ndir; ++i){
          for (int j=0; j <ndir; ++j){
            if (i==j){
              deltaij=1;
             }
            DSai(i,j)=2*Gmat*DEai(i,j)+lambdaMat*DEv*deltaij;
           } 
        } 

        }
        else if(tipoelem==2){
         //Chapa EPT
         DSai(0,0)=(2*Gmat/(1-vmat))*(DEai(0,0)+vmat*DEai(1,1));
         DSai(0,1)= 2*Gmat*DEai(0,1);
         DSai(1,0)=2*Gmat*DEai(1,0);
         DSai(1,1)=(2*Gmat/(1-vmat))*(DEai(1,1)+vmat*DEai(0,0));
          //std::cout << "DEai:\n" << DEai << std::endl;
          //std::cout << "DSai:\n" << DSai << std::endl;
        }
        else if(tipoelem==3){
         //Chapa EPD
         DSai(0,0)=kmat*((1-vmat)*DEai(0,0)+vmat*DEai(1,1));
         DSai(0,1)= 2*Gmat*DEai(0,1);
         DSai(1,0)=2*Gmat*DEai(1,0);
         DSai(1,1)=kmat*((1-vmat)*DEai(1,1)+vmat*DEai(0,0));

         }
        else if(tipoelem==4){

         }
        else if(tipoelem==5){

         }
        else if(tipoelem==6){

         }
        else {
         std::cout << "Tipo de elemento nao implementado." << std::endl;
        // return 50; //tipo de elemento nao implementado
        }


       //Calculo D2Eaibj
       D2Eaibj=0.5*(inversaA0.transpose()*DA1ai.transpose()*DA1bj*inversaA0+inversaA0.transpose()*DA1bj.transpose()*DA1ai*inversaA0);
       // trocar usando A total;


       //Calculo haibj
       double haibj=0.0; 
        for (int i=0; i <ndirSuperficie; ++i){
           for (int j=0; j <ndirSuperficie; ++j){ 
             haibj+=DEbj(i,j)*DSai(i,j)+Snl(i,j)*D2Eaibj(i,j);
           }
        }      
    //    PetscPrintf(PETSC_COMM_WORLD, "Matriz Hoz Antes:\n");
    //    MatView(Hoz, PETSC_VIEWER_STDOUT_WORLD);
        
       //Calculo e montagem Hessiana haibj :=> Hestatica
       int noa = propelemSuperficie(el, noelema); // Índices de array começam em 0 no C++
       int o = ndir * (noa - 1) + diri;
       int nob = propelemSuperficie(el, noelemb); // Índices de array começam em 0 no C++
       int z = ndir * (nob - 1) + dirj;
        //Hoz.coeffRef(o, z) += haibj*wihammer*J0; //Montando e integrando hessiana global - usando sparsa 
        //std::cout << "POS1: " << o << endl;std::cout << "POS2: " << z << endl;  
        MatSetValue(Hoz, o, z, haibj * wihammer * J0, ADD_VALUES);
       // MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
       // MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
        

    //    PetscPrintf(PETSC_COMM_WORLD, "Matriz Hoz deopois:\n");
    //    MatView(Hoz, PETSC_VIEWER_STDOUT_WORLD);


        double k= haibj*wihammer*J0; 
       // rows.push_back(o);
       // cols.push_back(z);
       // values.push_back(k);  //apagar

        //if (el == (1-1) && passotempo==1 ) {
        //if (el == (69-1) && diri==0 && noelema==0 && o==(1-1) && z==(20-1) && contDYkmax==10  ) {    
        //     // imprimir
        //     std::cout << "passotempo: " << passotempo << endl;  std::cout << "contDYkmax: " << contDYkmax << endl; 
        //     std::cout << "el: " << el << endl; std::cout << "ihammer: " << ihammer << endl; 
        //     std::cout << "A0: " << A0 << endl;std::cout << "A1: " << A1 << endl;std::cout << "Enl: " << Enl << endl;
        //     std::cout << "DA1ai: " << DA1ai << endl;
        //     std::cout << "DEai: " << DEai << endl;std::cout << "Snl: " << Snl << endl;
        //     std::cout << "fia: " << fia << endl; std::cout << "DSai: " << DSai << endl; std::cout << "D2Eaibj: " << D2Eaibj << endl;
        //     //std::cout << "Xil: " << Xil << endl;
        //     //std::cout << "ksi: " << ksi << endl;std::cout << "phi: " << phi << endl;std::cout << "dphi_dksi: " << dphi_dksi << endl;
        //      //std::cout << "DA1ai: " << DA1ai << endl;std::cout << "DA1bj: " << DA1bj << endl;std::cout << "A1: " << A1 << endl;
        //     //std::cout << "DEbj: " << DEbj << endl;std::cout << "DSai: " << DSai << endl;std::cout << "Snl: " << Snl << endl;std::cout << "D2Eaibj: " << D2Eaibj << endl;
        //     std::cout << "haibj: " << haibj << endl; std::cout << "wihammer: " << wihammer << endl;  std::cout << "J0: " << J0 << endl;
        //     std::cout << "POS1: " << o << endl;std::cout << "POS2: " << z << endl;std::cout << "H(o,z): " << Hoz.coeffRef(o, z) << endl;
        // }
        // double valorHoz=0.0;
        //  MatGetValues(Hoz, 1, &o, 1, &z, &valorHoz);
        // std::cout << "POS1: " << o << endl;std::cout << "POS2: " << z << endl;std::cout << "H(o,z): " << valorHoz << endl;
        //  std::cout << "haibj: " << haibj << endl; std::cout << "wihammer: " << wihammer << endl;  std::cout << "J0: " << J0 << endl;

        
          } //final for ndir j
        
          //return 0;
        } // final for nnoselem b
        //return 0;
      
    //final lacos globais da hessiana  
      } // final for ndir i
  
     } // final for nnoselem a
    //final lacos globais forca interna

    

    // Adiciona todos os valores a matriz com apenas uma chamada
   // MatSetValues(Hoz, rows.size(), rows.data(), cols.size(), cols.data(), values.data(), ADD_VALUES); //CONFERIR
    //ao adicionar valores a uma matriz esparsa, e necessário garantir que os indices estejam ordenados de maneira crescente

    
  
   // Integracao e Montagem do vetor de forças inerciais// Cálculo da Matriz de massa devido ao peso próprio
   // Matriz de massa muda quando modela otimização ou ELU
   rows.clear();
   cols.clear();
   values.clear();
   for (int a = 0; a < nnoselemSuperficie; ++a) {
    for (int i = 0; i < ndirSuperficie; ++i) {
   for (int y = 0; y < nnoselemSuperficie; ++y) {
    
     //A0,J0,phi, ja calculados para o elemento el
        double phialpha=phi(a);
        double phigama=phi(y);

        int noa = propelemSuperficie(el, a); 
        int noy= propelemSuperficie(el, y); 
        int pos1 = ndir * (noa - 1) + i;
        int pos2 = ndir * (noy - 1) + i;
    
        //Mp.coeffRef(pos1, pos2) += ro*phialpha*phigama*wihammer*J0; //Montando e integrando Matriz massa global - usando sparsa;
        //double valorMlocal=ro*phialpha*phigama*wihammer*J0;
        double k=ro*phialpha*phigama*wihammer*J0;
        rows.push_back(pos1);
        cols.push_back(pos2);
        values.push_back(k);
        MatSetValue(Mp, pos1, pos2, k, ADD_VALUES);
        
        // int poslocal1=ndir * (a) + i;
        // int poslocal2=ndir * (y) + i;
        //MmLocal(poslocal1,poslocal2)+=ro*phialpha*phigama*wihammer*J0; apagar
        //MmLocal(poslocal1,poslocal2)+=phialpha*phigama;
        
        // para 3D usar witetra; 1 ro para cada elemento;
      } // final for nnoselem y
     } //final for ndir i
   } // final for nnoselm a

    // Assemble matriz - usando o formato de montagem por blocos
    MatAssemblyBegin(Mp, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mp, MAT_FINAL_ASSEMBLY);
    
    // matriz de massa adicional estava aqui // apagar
    
    //std::cout << "ihammer: " << ihammer << endl; std::cout << "wihammer: " << wihammer << endl;std::cout << "ksi: " << ksi << endl; 
    //std::cout << "ro: " << ro << endl; std::cout << "wihammer: " << wihammer << endl;  std::cout << "J0: " << J0 << endl; 
    
} // final for pontos quadratura superficie wihammer


} //final for el // Final for GLOBAL elementos
// Assemble - usando o formato de montagem por blocos
//MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
//MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
 VecAssemblyBegin(Fjint);
 VecAssemblyEnd(Fjint);

 if (rank==0){
   std::cout << "*** Final laco for elementos ***" << std::endl;
 }





 /////////////////////////// Calculo da Matriz de massa total Mm: peso proprio + massa adicional ////////////////
   //Mm = Mp;
     //MatCopy(Mp, Mm, DIFFERENT_NONZERO_PATTERN);  // Copia os valores de Mp para Mm
     MatDuplicate(Mp, MAT_COPY_VALUES, &Mm); 
     //MatAssemblyBegin(Mm, MAT_FINAL_ASSEMBLY);
     //MatAssemblyEnd(Mm, MAT_FINAL_ASSEMBLY);

   for (int nocar = 0; nocar < nnoscar; ++nocar) {
     for (int i = 0; i < ndir; ++i) {
        // A massa é só 1 por nó, mas atua em todas as direções de 1 até ndir
        int noa = idcarganoSolido(nocar); 
        int pos = ndir * (noa - 1) + i;
        //Mm(pos, pos) += massaadc(nocar);
        //Mm.coeffRef(pos, pos) += massaadc(nocar); //operacao com matriz esparsa
        double k=massaadc(nocar);

        MatSetValue(Mm, pos, pos, k, ADD_VALUES); // montagem direta, fora do laco de integracao, pois adiciona-se apenas 1 vez na matriz final
      }
    }
    // Assemble - matriz Mm usando o formato de montagem por blocos
    MatAssemblyBegin(Mm, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mm, MAT_FINAL_ASSEMBLY);

     
     
  
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
 
rows.clear();
cols.clear();
values.clear();
//Contribuicao dos elementos de molas com graus de liberdade coincidentes: Hoz e Fjint 
for (int ap = 0; ap < naps; ++ap) {
    for (int dir = 0; dir < (ndirMolas-ndirMolasRotacao); ++dir) {
        double k = mola(ap, dir);
        int no = restrap(ap, 0); 
        int pos = ndir * (no - 1) + dir;

           //Fjint(pos) += k * (Yjf(pos) - Yjo(pos)); 
           PetscScalar kValue;
           VecGetValues(Yjf, 1, &pos, &kValue);
           PetscScalar lValue;
           VecGetValues(Yjo, 1, &pos, &lValue);
           VecSetValue(Fjint, pos, k * (kValue - lValue), ADD_VALUES);
          
          //Hoz.coeffRef(pos, pos) += k; //Molas com graus de liberdade coincidentes - matriz esparsa
          MatSetValue(Hoz, pos, pos, k, ADD_VALUES); 
       
        // rows.push_back(pos);
        // cols.push_back(pos);
        // values.push_back(k);
    }
}

// Assembly
MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
VecAssemblyBegin(Fjint);
VecAssemblyEnd(Fjint);



//Contribuicao el molas com graus de liberdade adicionais de rotacao
for (int m=0; m<nmolasRotacao; ++m){
    //k=molaR(m);
    //no=idmolaR(m,0);
    //dir=idmolaR(m,1);
    //pos=nnos*ndir+m;
    //Fjint(pos) += k * (Yjf(pos) - Yjo(pos)); 
    //Hoz.coeffRef(pos, pos) += k; 
}


// Cálculo da Matriz de amortecimento - Cc
 //Cc = ksi1Cc * Mm + ksi2Cc * Hoz; //C = ksi1 * Mm + ksi2 * Hoz;
     MatAXPY(Cc, ksi1Cc, Mm, DIFFERENT_NONZERO_PATTERN); // Adiciona * Mm a Cc
     MatAXPY(Cc, ksi2Cc, Hoz, DIFFERENT_NONZERO_PATTERN); // Adiciona *Hoz a Cc

// Se Analise Estatica; tipoanalise=1 estatica; 2 dinamica
if (tipoanalise == 1) {
    // Se Estatico, entao basta zerar Mm e Cc
    //Mm = Eigen::MatrixXd::Zero(nglt, nglt);C = Eigen::MatrixXd::Zero(nglt, nglt);
    // Mm = Eigen::SparseMatrix<double>(nglt, nglt); 
    // Cc = Eigen::SparseMatrix<double>(nglt, nglt); 
       MatZeroEntries(Mm);
       MatZeroEntries(Cc);
       // Finaliza a montagem das matrizes
        MatAssemblyBegin(Mm, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Mm, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(Cc, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Cc, MAT_FINAL_ASSEMBLY);
}

// Reservando Memoria Esparsa para Novas Setar Condicoes de Contorno sem Erro ////////////////////////////////////////////////////////////////////////
    for (int ap = 0; ap < naps; ++ap) {
        int no = restrap(ap, 0);
        for (int dir = 1; dir <= ndir; ++dir) {
            int gl = restrap(ap, dir); //Apoios_(r=1_impedido_r=0_livre_r=2_mola)_restrap_mola
            if (gl == 1) {
                int pos = ndir * (no - 1) + dir - 1;
                for (int vaux = 0; vaux < nglt; ++vaux) {
                    MatSetValue(Hd, pos, vaux, 0.0, INSERT_VALUES);
                    MatSetValue(Hd, vaux, pos, 0.0, INSERT_VALUES);
                }
                MatSetValue(Hd, pos, pos, 0.0, INSERT_VALUES);
            } 
        }
    }
     MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY);
     

// Calculo da Hessiana Dinamica /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Eigen::MatrixXd Hd = Hoz + (1.0 / (beta * Dt * Dt)) * Mm + (gama / (beta * Dt)) * Cc; //densa Newmark
//Eigen::SparseMatrix<double> Hd = alphaF*Hoz + (alphaM / (beta * Dt * Dt)) * Mm + (alphaF*gama / (beta * Dt)) * Cc; //esparsa alpha Generalizado
     
     MatAXPY(Hd, alphaF, Hoz, DIFFERENT_NONZERO_PATTERN); // Adiciona alphaF * Hoz a Hd
     MatAXPY(Hd, alphaM / (beta * Dt * Dt), Mm, DIFFERENT_NONZERO_PATTERN); // Adiciona (alphaM / (beta * Dt * Dt)) * Mm a Hd
     MatAXPY(Hd, alphaF * gama / (beta * Dt), Cc, DIFFERENT_NONZERO_PATTERN); // Adiciona (alphaF * gama / (beta * Dt)) * Cc a Hd

if (rank==0){


    std::cout << "***** Status 5: Hd montada. ********" << std::endl;
}

// Calculo da Primeira Aceleração; primeiro Qs, Rs /////////////////////////////////////////////////////////////////////////////////////
if (passotempo == 1 && contDYkmax == 1 && (tipoanalise != 1)) {
    // Resolvendo o sistema de equações usando Decomposição de Cholesky: matrizes simetricas e positivas definidas
    //Eigen::VectorXd Ajf = Eigen::LLT<Eigen::MatrixXd>(Mm).solve(Fjext - Fjint - Cc * Vjf); //densa

   // // Verificar se Mm é simétrica
   // if (!Mm.isApprox(Mm.transpose())) {
   //     std::cerr << "A matriz Mm não é simétrica: problema no calculo da primeira aceleracao: linha 1500+." << std::endl;
   //     return 1;
   // }

    // Verificar se Mm e simétrica
     PetscBool isSymmetric;
     MatIsSymmetric(Mm, 1.0e-2, &isSymmetric);
     if (!isSymmetric) {
         PetscPrintf(PETSC_COMM_WORLD, "A matriz Mm nao e simétrica: problema no calculo da primeira aceleração.\n");
         return 1;
     }

    // // Calcular a decomposição Cholesky de Mm
    // Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> cholesky(Mm);  

    // Verificar se a decomposição foi bem sucedida
    //if (cholesky.info() != Eigen::Success) {
    //    std::cerr << "A matriz Mm não é positiva definida: dentro calculo da primeira aceleracao,qs,rs: linha 1500+." << std::endl;
    //    return 1;
    //}

    // Resolver o sistema linear Ax = b usando a decomposição Cholesky
    //Eigen::VectorXd Ajf = cholesky.solve(Fjext - Fjint - Cc * Vjf);

    // Criar vetor b
     Vec b;
     VecDuplicate(Ajf, &b); // duplica apenas o formato

    // Calcular o lado direito da equação Ax = b
     VecCopy(Fjext, b);
     VecAXPY(b, -1.0, Fjint);
     
     Vec CcVjf;
     VecDuplicate(Qs, &CcVjf);
     MatMult(Cc, Vjf, CcVjf);
     VecAXPY(b, -1.0, CcVjf);

    // Definir tolerâncias
     double tolerancia_absoluta = 1.0e-16;
     double tolerancia_relativa = 1.0e-16;
     int max_iteracoes = 1000;

    // Configurar o solucionador GMRES
     KSP ksp;
     KSPCreate(PETSC_COMM_SELF, &ksp);
     //KSPSetType(ksp, KSPGMRES);
     KSPSetType(ksp, KSPCG);
     KSPSetOperators(ksp, Mm, Mm);
     KSPSetTolerances(ksp, tolerancia_absoluta, tolerancia_relativa, PETSC_DEFAULT, max_iteracoes);
     KSPSetFromOptions(ksp);
     KSPSetUp(ksp);

    // Resolver o sistema usando GMRES
    KSPSolve(ksp, b, Ajf);

    // Verificar sucesso da solução
         KSPConvergedReason reason;
         KSPGetConvergedReason(ksp, &reason);
         if (reason > 0) {
             //PetscPrintf(PETSC_COMM_SELF, "Convergiu com sucesso. Razão: %s\n", KSPConvergedReasons[reason]);
         } else if (reason < 0) {
             PetscPrintf(PETSC_COMM_SELF, "Divergiu no calculo As. Razao: %s\n", KSPConvergedReasons[reason]);
             return 99;
         } else {
             PetscPrintf(PETSC_COMM_SELF, "A solucao nao foi concluida no calculo As. Razao: %s\n", KSPConvergedReasons[reason]);
         }
    
    // Liberar memória
    VecDestroy(&b);
    VecDestroy(&CcVjf);
    KSPDestroy(&ksp);

       // Qs = Yjf / (beta * (Dt * Dt)) + Vjf / (beta * Dt) + ((1 / (2 * beta)) - 1) * Ajf;
       // Rs = Vjf + Dt * (1 - gama) * Ajf;
       // As = Ajf;
       // Pn=(1-alphaF)*(Fjint-Fjext)+(1-alphaM)*Mm*As+(1-alphaF)*Cc*Vs+(-alphaM*Mm-alphaF*gama*Dt*Cc)*Qs+alphaF*Cc*Rs;

    // Att
                // As
                VecCopy(Ajf, As); // no primeiro passo, As=Ajf e Ajf=As
                 // Calculo de Qs
                 VecCopy(Yjf, Qs);
                 VecScale(Qs, 1.0 / (beta * Dt * Dt));
                 VecAXPY(Qs, 1.0 / (beta * Dt),Vjf);
                 VecAXPY(Qs, ((1.0 / (2.0 * beta)) - 1.0), Ajf);
                
                // Calculo Rs
                 //Rs = Vjf + Dt * (1.0 - gama) * Ajf;
                 VecCopy(Vjf, Rs);
                 VecAXPY(Rs, Dt * (1.0 - gama),Ajf);

                // Calculo Pn
                 //Pn=(1.0-alphaF)*(Fjint-Fjext)+(1.0-alphaM)*Mm*As+(1.0-alphaF)*Cc*Vs+(-alphaM*Mm-alphaF*gama*Dt*Cc)*Qs+alphaF*Cc*Rs;
                 VecCopy(Fjint, Pn);
                 VecAXPY(Pn, -1.0,Fjext);
                 VecScale(Pn, (1.0-alphaF));

                 Mat MmCc;
                 MatDuplicate(Mm, MAT_COPY_VALUES, &MmCc);
                 MatScale(MmCc, -alphaM);
                 MatAXPY(MmCc, -alphaF*gama*Dt, Cc, DIFFERENT_NONZERO_PATTERN);

                 Vec MmAs, CcVs, MmCcQs, CcRs;
                 VecDuplicate(As, &MmAs); VecDuplicate(As, &CcVs); VecDuplicate(As, &MmCcQs); VecDuplicate(As, &CcRs);

                 MatMult(Mm, As, MmAs);
                 MatMult(Cc, Vs, CcVs);
                 MatMult(MmCc, Qs, MmCcQs);
                 MatMult(Cc, Rs, CcRs);

                 VecAXPY(Pn, (1.0-alphaM),MmAs);
                 VecAXPY(Pn, (1.0-alphaF), CcVs);
                 VecAXPY(Pn, 1.0, MmCcQs);
                 VecAXPY(Pn, alphaF, CcRs);

                 // Finaliza a montagem dos vetores
                  VecAssemblyBegin(Qs);
                  VecAssemblyEnd(Qs);
                  VecAssemblyBegin(Rs);
                  VecAssemblyEnd(Rs);
                  VecAssemblyBegin(Pn);
                  VecAssemblyEnd(Pn);

                  // Liberacao de memoria
                  MatDestroy(&MmCc);
                  VecDestroy(&MmAs); 
                  VecDestroy(&CcVs); 
                  VecDestroy(&MmCcQs); 
                  VecDestroy(&CcRs); 
}
 



// Calculo vetor de desbalanceamento mecanico (erro residual MNR) - Newmark ///////////////////////////////////////////////////////////////
  Vec gj; VecDuplicate(As, &gj); 

  VecCopy(Fjint, gj);
  VecAXPY(gj, -1.0,Fjext);
  VecScale(gj, alphaF);

  Mat MmCc;
  MatDuplicate(Mm, MAT_COPY_VALUES, &MmCc); // Mm
  MatScale(MmCc, (alphaM / (beta * (Dt * Dt)))); // ((alphaM*Mm / (beta * (Dt * Dt)))
  MatAXPY(MmCc, alphaF*gama / (beta * Dt), Cc, DIFFERENT_NONZERO_PATTERN);

  Vec MmCcYjf;
  VecDuplicate(As, &MmCcYjf);
  MatMult(MmCc, Yjf, MmCcYjf);

  VecAXPY(gj, 1.0, MmCcYjf);
  VecAXPY(gj, 1.0, Pn);
  
 // // finalizando montagem
 // VecAssemblyBegin(gj);
 // VecAssemblyEnd(gj);

  // Liberacao de memoria
     MatDestroy(&MmCc);
     VecDestroy(&MmCcYjf); 



// Aplicacao das condicoes de contorno em posicoes (Dirichlet) ////////////////////////////////////////////////////////////////////////
    for (int ap = 0; ap < naps; ++ap) {
        int no = restrap(ap, 0);
        for (int dir = 1; dir <= ndir; ++dir) {
            int gl = restrap(ap, dir); //Apoios_(r=1_impedido_r=0_livre_r=2_mola)_restrap_mola
            if (gl == 1) {
                int pos = ndir * (no - 1) + dir - 1;
                for (int vaux = 0; vaux < nglt; ++vaux) {
                    // Hd.coeffRef(pos, vaux) = 0.0;
                    // Hd.coeffRef(vaux, pos) = 0.0;
                    // Hoz.coeffRef(pos, vaux) = 0.0;
                    // Hoz.coeffRef(vaux, pos) = 0.0;
                    // gj(pos) = 0.0;
                    // if a posicao nao tem valor, nao faz nada, caso contrario mat set value

                    MatSetValue(Hd, pos, vaux, 0.0, INSERT_VALUES);
                    MatSetValue(Hd, vaux, pos, 0.0, INSERT_VALUES);
                    MatSetValue(Hoz, pos, vaux, 0.0, INSERT_VALUES);
                    MatSetValue(Hoz, vaux, pos, 0.0, INSERT_VALUES);
                    double valor=0.0;
                    VecSetValues(gj, 1, &pos, &valor, INSERT_VALUES);
                }
                // Hd.coeffRef(pos, pos) = 1.0;
                // Hoz.coeffRef(pos, pos) = 1.0;
                MatSetValue(Hd, pos, pos, 1.0, INSERT_VALUES);
                MatSetValue(Hoz, pos, pos, 1.0, INSERT_VALUES);
            } else if (gl == 2) {
                // Tratamento para molas (caso gl == 2)
                // nao ha restricoes adicionais, basta manter as contribucoes Fjint e Hd
            }
        }
    }

     MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY);
     MatAssemblyBegin(Hoz, MAT_FINAL_ASSEMBLY);
     MatAssemblyEnd(Hoz, MAT_FINAL_ASSEMBLY);
     VecAssemblyBegin(gj);
     VecAssemblyEnd(gj);

if (rank==0){


std::cout << " Status 8: gj e Hd montados e com condicoes de contorno aplicadas " << std::endl;
}
    
/////////////////////////////////////SOLVERS////////////////////////////////////////////////////////////////////////////
// Solucao do sistema de equacoes
    //Eigen::VectorXd DYk;
    // Criar vetor solucao DYk (correcao da posicao tentativa)
         Vec DYk, b;
         VecDuplicate(gj, &DYk); VecDuplicate(gj, &b); 
    // Verificação da consistencia e existência de inversa de Hd (matriz singular)
      PetscBool isSymmetric;
      MatIsSymmetric(Hoz, 1.0e-2, &isSymmetric);
      
      if (!isSymmetric) {
          PetscPrintf(PETSC_COMM_SELF, "A matriz Hoz não é simétrica.\n");
          return 0;
      }

      MatIsSymmetric(Hd, 1.0e-2, &isSymmetric);
      
      if (!isSymmetric) {
          PetscPrintf(PETSC_COMM_SELF, "A matriz Hd não é simétrica.\n");
          return 0;
      }
    
    //Solvers
    if (tipoSolver == 1) {
       // Mumps


    } else if (tipoSolver == 2) {
        // Solver iterativo
       
        
    } else if (tipoSolver == 3) {
    // Iterativo por Decomposicao Gradiente Conjugado
    
        // Montar vetor de desbalanceamento mecanico gj; HD * DYk = -gj ou  Ax = b
         VecCopy(gj, b);
         VecScale(b,-1.0); // b=-gj

         // Definir tolerâncias
          double tolerancia_absoluta = 1.0e-16;
          double tolerancia_relativa = 1.0e-16;
          int max_iteracoes = 1000;
    
        // Configurar o solucionador GMRES
         KSP ksp;
         KSPCreate(PETSC_COMM_SELF, &ksp);
         //KSPSetType(ksp, KSPGMRES);
         KSPSetType(ksp, KSPCG);
         KSPSetOperators(ksp, Hd, Hd);
         KSPSetTolerances(ksp, tolerancia_absoluta, tolerancia_relativa, PETSC_DEFAULT, max_iteracoes);

         // Configurar MUMPS como pré-condicionador
          PC pc;
          KSPGetPC(ksp, &pc);
          PCSetType(pc, PCLU);

          KSPSetFromOptions(ksp);
          KSPSetUp(ksp);
    
        // Resolver o sistema usando ksp escolhido
        KSPSolve(ksp, b, DYk);

        // Verificar sucesso da solução
         KSPConvergedReason reason;
         KSPGetConvergedReason(ksp, &reason);
         if (reason > 0) {
             //PetscPrintf(PETSC_COMM_SELF, "Convergiu com sucesso. Razão: %s\n", KSPConvergedReasons[reason]);
         } else if (reason < 0) {
             PetscPrintf(PETSC_COMM_SELF, "Divergiu. Razao: %s\n", KSPConvergedReasons[reason]);
             return 99;
         } else {
             PetscPrintf(PETSC_COMM_SELF, "A solucao nao foi concluida. Razao: %s\n", KSPConvergedReasons[reason]);
         }
        
        // Liberar memória
        VecDestroy(&b);
        KSPDestroy(&ksp);

     }   else {
        std::cerr << "Tipo de solver inválido: solucao Hd." << std::endl;
        return 1;
    }    
   
    //DYkmax = DYk.lpNorm<Eigen::Infinity>();
    // Calculo do residuo maximo DYKmax
       DYkmax;
      VecNorm(DYk, NORM_INFINITY, &DYkmax);
      
      PetscPrintf(PETSC_COMM_SELF, "passo: %d\n", passotempo);
      PetscPrintf(PETSC_COMM_SELF, "contDYkmax: %d\n", contDYkmax);
      PetscPrintf(PETSC_COMM_SELF, "DYkmax: %f\n", DYkmax);

    


    //Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    //solver.compute(Hd);
    //if (solver.info() != Eigen::Success) {
    //    std::cerr << "Erro na decomposição LU." << std::endl;
    //    return 1;
    //}
    //double determinant = solver.determinant();
    //std::cout << "Determinante da matriz: " << determinant << std::endl;
///////////////////////////////////// SOLVERS ////////////////////////////////////////////////////////////////////////////

if (rank==0){
std::cout << "Status  9: sistema HD*DY=-gj resolvido para DY " << std::endl;
}
 
// Correcao da posicao atual para nova posicao tentativa
VecAXPY(Yjf, 1.0, DYk);

// Correção da Velocidade Atual e Aceleração Atual
if (!(tipoanalise == 1)) {

    VecCopy(Yjf,Vjf); VecScale(Vjf, (gama / (beta * Dt)));
    VecAXPY(Vjf, 1.0, Rs);
    VecAXPY(Vjf, - gama * Dt, Qs);

    VecCopy(Yjf,Ajf); VecScale(Ajf, 1.0 / (beta * Dt * Dt));
    VecAXPY(Ajf, -1.0, Qs);

}


// liberar memoria
    //VecDestroy(&DYk);


//final laco while DYKmax - MNR   
//convergencia do nivel carga atual
//} // Final laco global While MNR - POSICAO EQUILIBRIO DYKMAX - NEWTON RAPHSON
///////////////////// PARTE FINAL DO SOLIDO DENTRO DE MNR //////////////////////////////////////////////////////////////////////////




///////////////////// // Acoplamento Geometrico //////////////////////////////////////////////////////////////////////////////////////////////////////////

            //Aplicar deslocamentos atuais do solido no contorno FE para a malha do fluido (Dirichlet)
            if (atualizarPosicaoMalha){

            // atualizar velocidade no fluido com Velocidade_Solido (Direchlet no fluido)
            for (int no=0; no<numBoundNodes; ++no){
                int noSolido = conectIFE(no,0);
                int noFluido = conectIFE(no,1);

                // pegar posicao no solido
                double y1 = Yjf(noSolido*2);
                double y2 = Yjf(noSolido*2+1);

                // buscar elemento relativo ao noFluido
                // verificar rank MPI
               // if (rank_elem==rank){
                    //Atualizacao das coordenadas
                    typename Node::VecLocD v;
                      x = nodes_[noFluido]->getCoordinates();
                      x(0) = y1; // substituicao 
                      x(1) = y2;
                      nodes_[noFluido] -> setUpdatedCoordinates(x);  //nodes_[noFluido] -> setUpdatedVelocity(v);
                //}
            }
            }

    /*
      // acoplamento antigo///////////////////////////////////////////////
       for (int i=0; i < numBoundElems; i++){
           if (boundary_[i] -> getConstrain(0) == 3 || boundary_[i] -> getConstrain(1) == 3 ){
           //if (boundary_[i] -> getConstrain(0) == 3 ){
               //std::cout << "asasa " << i << std::endl;
               Boundaries::BoundConnect connectB;
               connectB = boundary_[i] -> getBoundaryConnectivity();
               int no1 = connectB(0); // no 1 do elemento i de linha do contorno
               int no2 = connectB(1); // no 2 do elemento i de linha do contorno
               int no3 = connectB(2); // no 3 do elemento i de linha do contorno

               typename Node::VecLocD x;
               
               //Atualizacao das coordenadas X e Y
               x = nodes_[no1]->getCoordinates();
               x(0) += .1 * dTime;
               x(1) += .1 * dTime;
               nodes_[no1] -> setUpdatedCoordinates(x);

               x = nodes_[no2]->getCoordinates();
                     // std::cout << "elemento: " << i << std::endl;
                     // std::cout << "Coord X Anterior " << x(0) << std::endl;
               x(0) += .1 * dTime;
               x(1) += .1 * dTime;
               nodes_[no2] -> setUpdatedCoordinates(x);
                     // std::cout << "Coord X atual " << nodes_[no2]->getUpdatedCoordinates()(0) << std::endl;
                     // std::cout << "\n " << std::endl;
              
        
               x = nodes_[no3]->getCoordinates();
               x(0) += .1 * dTime;
               x(1) += .1 * dTime;
               nodes_[no3] -> setUpdatedCoordinates(x);


               //Atualizacao das velocidades do fluido em X e Y

           };
       };
       */

        // movimentacao da malha pela equacao de Laplace
        solveSteadyLaplaceProblem(1, 1.e-6); // dentro while MNR ok
        // acoplamento ///////////////////////////////////////////////

        //-> correção correção Y_malha: coordenadas ja sao atualizadas dentro de solveSteadyLaplaceProblem: ok

        // -> calculo velo_malha usando a-generalizado;
       
        // -> resíduo=normaL2(resíduoFluido, resíduoSólido);



            //Computes the solution vector norm
            //ierr = VecNorm(u,NORM_2,&val);CHKERRQ(ierr);
   
            boost::posix_time::ptime t2 =                               \
                               boost::posix_time::microsec_clock::local_time();
         
            if(rank == 0){
                boost::posix_time::time_duration diff = t2 - t1;

                std::cout << "Iteration = " << inewton 
                          << " (" << iterations << ")"  
                          << "   Du Norm = " << std::scientific << sqrt(duNorm) 
                          << " " << sqrt(dpNorm)
                          << "  Time (s) = " << std::fixed
                          << diff.total_milliseconds()/1000. << std::endl;
            };
                      
            ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
            ierr = VecDestroy(&b); CHKERRQ(ierr);
            ierr = VecDestroy(&u); CHKERRQ(ierr);
            ierr = VecDestroy(&All); CHKERRQ(ierr);
            ierr = MatDestroy(&A); CHKERRQ(ierr);
           
           if (DYkmax < tol){
              // if (rank==0){
              //     std::cout << "Convergiu para o solido = " << inewton 
              //           << "   DY Norm = " << std::scientific << DYkmax << std::endl;
              // }
              if ((iTimeStep>2)&&(sqrt(duNorm) <= tolerance)) {
                break;
               };
           }
            

            
        };//Newton-Raphson
//////////////////////LACO WHILE - SISTEMA NAO LINEAR - TODOS 3 SISTEMAS ////////////////////////////////////////////////////////////////////////////////////////////

if (rank==0){
    std::cout << "passotempo:" << passotempo << endl; 
std::cout << "**contDYkmax:" << endl << contDYkmax << endl;
std::cout << "**DYkmax:" << std::scientific << DYkmax << endl;
std::cout << "**********Laco Global While - Metodo Newton Raphson - Finalizado com Sucesso**********" << std::endl; 

}

// Deslocamentos não lineares para o nível de carga atual
//Eigen::VectorXd Unl = Yjf - Yjo;
Vec Unl;
VecDuplicate(Yjf, &Unl);
VecCopy(Yjf, Unl);
VecAXPY(Unl, -1.0, Yjo);



 
        // Compute and print drag and lift coefficients
        if (computeDragAndLift){
            dragAndLiftCoefficients(dragLift);
        };
        return 0;


        //Printing results
        printResults(iTimeStep);

        





//////// IMPRESSAO SOLIDO /////////////////////////

//Area para impressao dos dados para cada passo de tempo
// Verifica se o arquivo está aberto
if (!outputFile.is_open()) {
    std::cout << "Erro: O arquivo de saída não está aberto para dados a cada passo de tempo." << std::endl;
    return 1; // Encerra o programa com erro
}

//Impressao ou registro da posicao e do nivel de carga tempo t passotempo
if (iapc == 1) {
    // Impressao do cabeçalho
    outputFile << "***********************************************************"
               << "****************" << std::endl;
    outputFile << "\nDeslocamentos Nodais para cada passo de Carga:\n";
    outputFile << "tempo = " << std::scientific << t << "\n";
    outputFile << "\ngl         Unl           FjextAtual\n";

    // Configurando a saída para ter uma precisão fixa com 6 casas decimais
    outputFile << std::fixed;
    outputFile.precision(6);

    for (int gl = 0; gl < nglt; ++gl) {
        outputFile.width(3); //configura largura

        double unl_value, fjext_value;
        VecGetValues(Unl, 1, &gl, &unl_value);
        VecGetValues(Fjext, 1, &gl, &fjext_value);

        outputFile << gl + 1;
        outputFile << "     " << unl_value;
        outputFile << "     " << fjext_value;
        outputFile << "\n";
    }
}

// Impressao vetor de deslocamentos nodais 
if (iapc == 2) {
    if (t == 0) {
        // Impressao do cabeçalho
        outputFile << "***********************************************************"
                   << "****************" << std::endl;
        outputFile << "\nDeslocamentos Nodais para cada passo de Tempo:\n";
        outputFile << "gl         U\n";
    }
    outputFile << "t = " << std::scientific << t << "\n";

    for (int gl = 0; gl < nglt; ++gl) {
        outputFile.width(3);

        double unl_value;
        VecGetValues(Unl, 1, &gl, &unl_value);

        outputFile << gl + 1;
        outputFile << "     " << unl_value;
        outputFile << "\n";
    }
}

// Impressao das posicoes nodais para cada passo de tempo de trelicas
if (iapc == 3) {
    if (t == 0) {
        // Impressao do cabeçalho
        outputFile << "***********************************************************"
                   << "****************" << std::endl;
        outputFile << "\nPosicoes Nodais para cada passo de Tempo:\n";
        outputFile << "no         X         Y         Z\n";
    }
    outputFile << "t = " << std::scientific << t << "\n";

    for (int no = 1; no <= nnos; ++no) {
        int glx = ndir * (no - 1) + 1;
        int gly = ndir * (no - 1) + 2;
        int glz = ndir * (no - 1) + 3;
        outputFile.width(3);

        double yjf_x, yjf_y, yjf_z;
        VecGetValues(Yjf, 1, &glx, &yjf_x); 
        VecGetValues(Yjf, 1, &gly, &yjf_y);
        VecGetValues(Yjf, 1, &glz, &yjf_z);

        outputFile << no;
        outputFile << "     " << yjf_x;
        outputFile << "     " << yjf_y;
        outputFile << "     " << yjf_z;
        outputFile << "\n";
    }
}

// Impressao ParaView
std::string item = std::to_string(passotempo); //converte int para string
std::string aux = "saidapos" + item + ".vtu";

std::ofstream fid(aux); // Abre o arquivo saidaposX.vtu para escrita

// Escreve as informações de cabeçalho no arquivo
fid << "<?xml version=\"1.0\"?>\n";
fid << "<VTKFile type=\"UnstructuredGrid\">\n";
fid << "  <UnstructuredGrid>\n";
fid << "  <Piece NumberOfPoints=\"" << nnos << "\" NumberOfCells=\"" << nelems << "\">\n";

// Escreve as coordenadas nodais no arquivo
fid << "    <Points>\n";
fid << "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
for (int no = 0; no < nnos; ++no) {
    fid << "        " << coordno(no, 0) << " " << coordno(no, 1) << " " << coordno(no, 2) << "\n";
}
fid << "      </DataArray>\n";
fid << "    </Points>\n";


// Escreve a conectividade dos elementos no arquivo
fid << "    <Cells>\n";
fid << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
for (int j = 0; j < nelems; ++j) {
    for (int k = 0; k < nnoselem; ++k) {
        fid << " " << propelem(j, k) - 1; // Subtrai 1 para ajustar a base para zero; quando usar Gmsh deve tirar isso
    }
    fid << "\n";
}
fid << "      </DataArray>\n";


// Escreve os offsets no arquivo
fid << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
for (int jel = 0; jel < nelems; ++jel) {
    fid << "        " << (jel + 1) * nnoselem << "\n";
}
fid << "      </DataArray>\n";

// Escreve os tipos de células/elementos no arquivo
fid << "      <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
for (int jel = 0; jel < nelems; ++jel) {
    fid << "        69\n";  //"3"  a triângulos. "5"  a quadriláteros. "10"  a tetraedros. "12"  a hexaedros
}
fid << "      </DataArray>\n";

fid << "    </Cells>\n";


// Resultados nodais
// Escreve os deslocamentos nodais no arquivo
fid << "    <PointData>\n";
fid << "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"Displac\" format=\"ascii\">\n";
for (int no = 0; no < nnos; ++no) {
        double unl_1, unl_2, unl_3;
        int pos1, pos2, pos3;
        pos1=ndir * no;
        pos2=ndir * no + 1;
        pos3=ndir * no + 2;

        VecGetValues(Unl, 1, &pos1, &unl_1); 
        VecGetValues(Unl, 1, &pos2, &unl_2);
        
    fid << "        " << unl_1 << " " << unl_2;

    if (ndir == 2) {
        fid << " 0.0"; // Terceira coordenada igual a zero para ndir=2
    } else if (ndir == 3) {
        VecGetValues(Unl, 1, &pos3, &unl_3);
        fid << " " << unl_3; // Terceira coordenada para ndir=3
    }
    
    fid << "\n";
}

// Escreve as velocidades nodais no arquivo
fid << "      </DataArray>\n";
fid << "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"Velocidade\" format=\"ascii\">\n";
for (int no = 0; no < nnos; ++no) {
        double vnl_1, vnl_2, vnl_3;
        int pos1, pos2, pos3;
        pos1=ndir * no;
        pos2=ndir * no + 1;
        pos3=ndir * no + 2;

        VecGetValues(Vjf, 1, &pos1, &vnl_1); 
        VecGetValues(Vjf, 1, &pos2, &vnl_2);

    fid << "        " << vnl_1 << " " << vnl_2;

    if (ndir == 2) {
        fid << " 0.0"; // Terceira coordenada igual a zero para ndir=2
    } else if (ndir == 3) {
        VecGetValues(Vjf, 1, &pos3, &vnl_3);
        fid << " " << vnl_3; // Terceira coordenada para ndir=3
    }
    
    fid << "\n";
}

// Escreve as aceleracoes nodais no arquivo
fid << "      </DataArray>\n";
fid << "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"Aceleracao\" format=\"ascii\">\n";
for (int no = 0; no < nnos; ++no) {
        double anl_1, anl_2, anl_3;
        int pos1, pos2, pos3;
        pos1=ndir * no;
        pos2=ndir * no + 1;
        pos3=ndir * no + 2;

        VecGetValues(Ajf, 1, &pos1, &anl_1); 
        VecGetValues(Ajf, 1, &pos2, &anl_2);

    fid << "        " << anl_1 << " " << anl_2;

    if (ndir == 2) {
        fid << " 0.0"; // Terceira coordenada igual a zero para ndir=2
    } else if (ndir == 3) {
        VecGetValues(Ajf, 1, &pos3, &anl_3);
        fid << " " << anl_3; // Terceira coordenada para ndir=3
    }
    
    fid << "\n";
}

// Escreve as posicoes nodais no arquivo
fid << "      </DataArray>\n";
fid << "      <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"Posicao\" format=\"ascii\">\n";
for (int no = 0; no < nnos; ++no) {
        double ynl_1, ynl_2, ynl_3;
        int pos1, pos2, pos3;
        pos1=ndir * no;
        pos2=ndir * no + 1;
        pos3=ndir * no + 2;

        VecGetValues(Yjf, 1, &pos1, &ynl_1); 
        VecGetValues(Yjf, 1, &pos2, &ynl_2);

    fid << "        " << ynl_1 << " " << ynl_2;

    if (ndir == 2) {
        fid << " 0.0"; // Terceira coordenada igual a zero para ndir=2
    } else if (ndir == 3) {
        VecGetValues(Yjf, 1, &pos3, &ynl_3);

        fid << " " << ynl_3; // Terceira coordenada para ndir=3
    }
    
    fid << "\n";
}
fid << "      </DataArray>\n";
fid << "    </PointData>\n";

// Resultados nos Elementos
fid << "    <CellData>" << "\n";
   // Implementar tensoes/deformacoes
fid << "    </CellData>" << "\n";



fid << "  </Piece>\n";
fid << "  </UnstructuredGrid>\n";
fid << "</VTKFile>\n";

fid.close(); // Fecha o arquivo paraView saidaposX.vtu

   
    //final laco for temporal Newmark
//} // Final for tempo - LACO GLOBAL TEMPORAL - METODO NEWMARK

std::cout << "**********Laco Global Temporal - Metodo Newmark - Finalizado com Sucesso**********" << std::endl;  
//////////////////////LACO FOR - INTEGRACAO TEMPORAL ////////////////////////////////////////////////////////////////////////////////////////////  


///// IMPRESSAO SOLIDO ACIMA ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  




    }; // final for temporal dos 3 sistemas






/////////////////////CONFIGURACAO POS PROCESSAMENTO //////////////////////////////////////////////////////////////////////////////////////////// 
    //FUNCAO PARA IMPRESSAO E RESULTADOS FINAIS NO ARQUIVO saida_pos.txt

    // Arquivo aberto no inicio 
    //std::ofstream outputFile("saida_dados.txt"); //arquivo aberto na funcao dinamica
    if (!outputFile.is_open()) {
        std::cerr << "Arquivo nao esta aberto na impressao dos resultados finais." << std::endl;
        return 1; // saida do programa com codigo de erro
    }

    // Configurando a saida para ter uma precisão fixa
    outputFile << std::fixed;
    outputFile.precision(6);

    // Impressao dos escalares no arquivo de saída
    outputFile << "Analise Dinamica Método dos Elementos Finitos Posicional - Escalares:" << std::endl;
    outputFile << "tipoelem: " << tipoelem << std::endl;
    outputFile << "nmats: " << nmats << std::endl;
    outputFile << "nelems: " << nelems << std::endl;
    outputFile << "naps: " << naps << std::endl;
    outputFile << "nmolas: " << nmolas << std::endl;
    outputFile << "nmn: " << nmn << std::endl;
    outputFile << "nnoscar: " << nnoscar << std::endl;
    outputFile << "tipoanalise: " << tipoanalise << std::endl;
    outputFile << "npassosCarga: " << npassosCarga << std::endl;
    outputFile << "npassostempo: " << npassostempo << std::endl;
    outputFile << "residuoadm: " << residuoadm << std::endl;
    outputFile << "tt: " << tt << std::endl;

    // Impressão das matrizes no arquivo de saída
    outputFile << "\nMatrizes:**************************************" << std::endl;
    outputFile << "Matriz coordno:" << std::endl << coordno << std::endl;
    outputFile << "Matriz propmat:" << std::endl << propmat << std::endl;
    outputFile << "Matriz propelem:" << std::endl << propelem << std::endl;
    outputFile << "Matriz restrap:" << std::endl << restrap << std::endl;
    outputFile << "Matriz mola:" << std::endl << mola << std::endl;
    outputFile << "Matriz idcarganoSolido:" << std::endl << idcarganoSolido << std::endl;
    outputFile << "Matriz campodeslo:" << std::endl << campodeslo << std::endl;
    outputFile << "Matriz massa adicional:" << std::endl << massaadc << std::endl;
    outputFile << "Matriz carganoSolido:" << std::endl << carganoSolido << std::endl;
 
    
    outputFile.close(); // Fechamento do arquivo de saída principa saida_dados.txt
     std::cout << "alphaF:" << std::endl << alphaF << std::endl;
     std::cout << "alphaM:" << std::endl << alphaM << std::endl;
    std::cout << "**********Analise Dinamica Completa Finalizada com Sucesso**********" << std::endl;
    //////////////////////CONFIGURACAO POS PROCESSAMENTO ////////////////////////////////////////////////////////////////////////////////////////////

    //destruir os vetores no final para evitar vazamento de memória.
    VecDestroy(&Vjo);
    VecDestroy(&Ajo);
    VecDestroy(&Vjf);
    VecDestroy(&Ajf);
    VecDestroy(&Qs);
    VecDestroy(&Rs);
    VecDestroy(&Pn);
    VecDestroy(&Ujrecal);
    VecDestroy(&Yjo);
    VecDestroy(&Yjf);
    VecDestroy(&Fjint);
    VecDestroy(&Fjext); 
    VecDestroy(&Ys); 
    VecDestroy(&Vs); 
    VecDestroy(&As); 


   

    // Libere recursos
    MatDestroy(&Hoz);MatDestroy(&Hd);MatDestroy(&Mm);MatDestroy(&Mp);MatDestroy(&Cc);
    
    PetscFinalize();
    std::cout << "======>>**********Todas funcoes Solido Finalizadas com Sucesso**********" << std::endl;
    //return 0; // programa finalizado com sucesso
//} final main solido






if (rank==0){
    std::cout << "****************** status Atual: ok *********************\n";
}

return 0;




    
    return 0;
};

//------------------------------------------------------------------------------
//-------------------------SOLVE TRANSIENT FLUID PROBLEM------------------------
//------------------------------------------------------------------------------
template<>
int Fluid<2>::solveTransientProblemMoving(int iterNumber, double tolerance) {

    Mat               A;
    Vec               b, u, All;
    PetscErrorCode    ierr;
    PetscInt          Istart, Iend, Ii, Ione, iterations;
    KSP               ksp;
    PC                pc;
    VecScatter        ctx;
    PetscScalar       val;
    //    MatNullSpace      nullsp;
    int rank;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::ofstream dragLift;
    dragLift.open("dragLift.dat", std::ofstream::out | std::ofstream::app);
    if (rank == 0) {
        dragLift << "Time   Pressure Drag   Pressure Lift " 
                 << "Friction Drag  Friction Lift Drag    Lift " 
                 << std::endl;
    };    

    // Set element mesh moving parameters
    double vMax = 0., vMin = 1.e10;
    for (int i = 0; i < numElem; i++){
        double v = elements_[i] -> getJacobian();
        if (v > vMax) vMax = v;
        if (v < vMin) vMin = v;
    };
    for (int i = 0; i < numElem; i++){
        double v = elements_[i] -> getJacobian();
        double eta = 1 + (1. - vMin / vMax) / (v / vMax);
        elements_[i] -> setMeshMovingParameter(eta); // ver na tese do jefferson
    };

    iTimeStep = 0.;

    for (iTimeStep = 0; iTimeStep < numTimeSteps; iTimeStep++){

        for (int i = 0; i < numElem; i++){
            elements_[i] -> getParameterSUPG();
        };

        
        if (rank == 0) {std::cout << "------------------------- TIME STEP = "
                                  << iTimeStep << " -------------------------"
                                  << std::endl;}
        
        for (int i = 0; i < numNodes; i++){
            double accel[2], u[2], uprev[2];
            
            //Compute acceleration
            u[0] = nodes_[i] -> getVelocity(0);
            u[1] = nodes_[i] -> getVelocity(1);
            
            uprev[0] = nodes_[i] -> getPreviousVelocity(0);
            uprev[1] = nodes_[i] -> getPreviousVelocity(1);
            
            accel[0] = (u[0] - uprev[0]) / dTime;
            accel[1] = (u[1] - uprev[1]) / dTime;
            
            nodes_[i] -> setAcceleration(accel);
            
            //Updates velocity
            nodes_[i] -> setPreviousVelocity(u);
        };


        // Moving boundary
        for (int i = 0; i < numNodes; i++){
            typename Node::VecLocD x;
            
            x = nodes_[i] -> getCoordinates();
            nodes_[i] -> setPreviousCoordinates(0,x(0));
            nodes_[i] -> setPreviousCoordinates(1,x(1));
        };

        //for (int i=0; i < numBoundElems; i++){
        //    if (boundary_[i] -> getConstrain(0) == 3){
        //        //std::cout << "asasa " << i << std::endl;
        //        Boundaries::BoundConnect connectB;
        //        connectB = boundary_[i] -> getBoundaryConnectivity();
        //        int no1 = connectB(0);
        //        int no2 = connectB(1);
        //        int no3 = connectB(2);

         //       typename Node::VecLocD x;
         //       x = nodes_[no1]->getCoordinates();
         //       x(1) -= .1 * dTime*0;
         //       nodes_[no1] -> setUpdatedCoordinates(x);

         //       x = nodes_[no2]->getCoordinates();
         //       x(1) -= .1 * dTime*0;
         //       nodes_[no2] -> setUpdatedCoordinates(x);
         
         //       x = nodes_[no3]->getCoordinates();
           //     x(1) -= .1 * dTime*0;
             //   nodes_[no3] -> setUpdatedCoordinates(x);
            //};
        //};

        solveSteadyLaplaceProblem(1, 1.e-6);

        for (int i=0; i< numNodes; i++){
            typename Node::VecLocD x,xp;
            double u[2];
            
            x = nodes_[i] -> getCoordinates();
            xp = nodes_[i] -> getPreviousCoordinates();
            
            u[0] = (x(0) - xp(0)) / dTime;
            u[1] = (x(1) - xp(1)) / dTime;

            nodes_[i] -> setMeshVelocity(u);
        };


        double duNorm=100.;
        
        for (int inewton = 0; inewton < iterNumber; inewton++){
            boost::posix_time::ptime t1 =                             
                               boost::posix_time::microsec_clock::local_time();
            
            ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                                2*numNodes+numNodes, 2*numNodes+numNodes,
                                100,NULL,300,NULL,&A); 
            CHKERRQ(ierr);
            
            ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);
            
            //Create PETSc vectors
            ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
            ierr = VecSetSizes(b,PETSC_DECIDE,2*numNodes+numNodes);
            CHKERRQ(ierr);
            ierr = VecSetFromOptions(b);CHKERRQ(ierr);
            ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
            ierr = VecDuplicate(b,&All);CHKERRQ(ierr);
            
            for (int jel = 0; jel < numElem; jel++){   
                
                if (part_elem[jel] == rank) {
                    //Compute Element matrix
                    elements_[jel] -> getTransientNavierStokes();

                    typename Elements::LocalMatrix Ajac;
                    typename Elements::LocalVector Rhs;
                    typename Elements::Connectivity connec;
                    
                    //Gets element connectivity, jacobian and rhs 
                    connec = elements_[jel] -> getConnectivity();
                    Ajac = elements_[jel] -> getJacNRMatrix();
                    Rhs = elements_[jel] -> getRhsVector();
                    
                    //Disperse local contributions into the global matrix
                    //Matrix K and C
                    for (int i=0; i<6; i++){
                        for (int j=0; j<6; j++){
                            if (fabs(Ajac(2*i  ,2*j  )) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * connec(j);
                                ierr = MatSetValues(A, 1, &dof_i,1, &dof_j,
                                                    &Ajac(2*i  ,2*j  ),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i+1,2*j  )) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i+1,2*j  ),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i  ,2*j+1)) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * connec(j) + 1;
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i  ,2*j+1),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i+1,2*j+1)) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * connec(j) + 1;
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i+1,2*j+1),
                                                    ADD_VALUES);
                            };
                        
                            //Matrix Q and Qt
                            if (fabs(Ajac(2*i  ,12+j)) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i  ,12+j),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(12+j,2*i  )) >= 1.e-15){
                                int dof_i = 2 * connec(i);
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_j, 1, &dof_i,
                                                    &Ajac(12+j,2*i  ),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(2*i+1,12+j)) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(2*i+1,12+j),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(12+j,2*i+1)) >= 1.e-15){
                                int dof_i = 2 * connec(i) + 1;
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_j, 1, &dof_i,
                                                    &Ajac(12+j,2*i+1),
                                                    ADD_VALUES);
                            };
                            if (fabs(Ajac(12+i,12+j)) >= 1.e-15){
                                int dof_i = 2 * numNodes + connec(i);
                                int dof_j = 2 * numNodes + connec(j);
                                ierr = MatSetValues(A, 1, &dof_i, 1, &dof_j,
                                                    &Ajac(12+i,12+j),
                                                    ADD_VALUES);
                            };
                        };
                        
                        //Rhs vector
                        if (fabs(Rhs(2*i  )) >= 1.e-15){
                            int dof_i = 2 * connec(i);
                            ierr = VecSetValues(b, 1, &dof_i, &Rhs(2*i  ),
                                                ADD_VALUES);
                        };
                        
                        if (fabs(Rhs(2*i+1)) >= 1.e-15){
                            int dof_i = 2 * connec(i)+1;
                            ierr = VecSetValues(b, 1, &dof_i, &Rhs(2*i+1),
                                                ADD_VALUES);
                        };
                        if (fabs(Rhs(12+i)) >= 1.e-15){
                            int dof_i = 2 * numNodes + connec(i);
                            ierr = VecSetValues(b, 1, &dof_i, &Rhs(12+i),
                                                ADD_VALUES);
                        };
                    };
                };
            }; //Elements
            
            //Assemble matrices and vectors
            ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            
            ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
            ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
            
            // MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            // ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            
            //Create KSP context to solve the linear system
            ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
            
            ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
            



            // ierr = KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,
            //                         500);CHKERRQ(ierr);
            
            //ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
            
            // ierr = KSPGetPC(ksp,&pc);
            
            // ierr = PCSetType(pc,PCJACOBI);
            
            // ierr = KSPSetType(ksp,KSPDGMRES); CHKERRQ(ierr);

            //ierr = KSPGMRESSetRestart(ksp, 500); CHKERRQ(ierr);
            
            // //    ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);
            

            // ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nullsp);
            // ierr = MatSetNullSpace(A, nullsp);
            // ierr = MatNullSpaceDestroy(&nullsp);

 

#if defined(PETSC_HAVE_MUMPS)
            ierr = KSPSetType(ksp,KSPPREONLY);
            ierr = KSPGetPC(ksp,&pc);
            ierr = PCSetType(pc, PCLU);
            //      MatMumpsSetIcntl(A,25,-1);
#endif          
            ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
            ierr = KSPSetUp(ksp);


            ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

            ierr = KSPGetTotalIterations(ksp, &iterations);            

            //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);CHKERRQ(ierr);
            
            //Gathers the solution vector to the master process
            ierr = VecScatterCreateToAll(u, &ctx, &All);CHKERRQ(ierr);
            
            ierr = VecScatterBegin(ctx, u, All, INSERT_VALUES, SCATTER_FORWARD);
            CHKERRQ(ierr);
            
            ierr = VecScatterEnd(ctx, u, All, INSERT_VALUES, SCATTER_FORWARD);
            CHKERRQ(ierr);
            
            ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
            
            //Updates nodal values
            double p_;
            duNorm = 0.;
            double dpNorm = 0.;
            Ione = 1;
            
            for (int i = 0; i < numNodes; ++i){
                //if (nodes_[i] -> getConstrains(0) == 0){
                    Ii = 2*i;
                    ierr = VecGetValues(All, Ione, &Ii, &val);CHKERRQ(ierr);
                    duNorm += val*val;
                    nodes_[i] -> incrementVelocity(0,val);
                    //}; 
                
                    //if (nodes_[i] -> getConstrains(1) == 0){
                    Ii = 2*i+1;
                    ierr = VecGetValues(All, Ione, &Ii, &val);CHKERRQ(ierr);
                    duNorm += val*val;
                    nodes_[i] -> incrementVelocity(1,val);
                    //};
            };
            
            for (int i = 0; i<numNodes; i++){
                Ii = 2*numNodes+i;
                ierr = VecGetValues(All,Ione,&Ii,&val);CHKERRQ(ierr);
                p_ = val;
                dpNorm += val*val;
                nodes_[i] -> incrementPressure(p_);
            };
            
            //Computes the solution vector norm
            //ierr = VecNorm(u,NORM_2,&val);CHKERRQ(ierr);
   
            boost::posix_time::ptime t2 =                               \
                               boost::posix_time::microsec_clock::local_time();
         
            if(rank == 0){
                boost::posix_time::time_duration diff = t2 - t1;

                std::cout << "Iteration = " << inewton 
                          << " (" << iterations << ")"  
                          << "   Du Norm = " << std::scientific << sqrt(duNorm) 
                          << " " << sqrt(dpNorm)
                          << "  Time (s) = " << std::fixed
                          << diff.total_milliseconds()/1000. << std::endl;
            };
                      
            ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
            ierr = VecDestroy(&b); CHKERRQ(ierr);
            ierr = VecDestroy(&u); CHKERRQ(ierr);
            ierr = VecDestroy(&All); CHKERRQ(ierr);
            ierr = MatDestroy(&A); CHKERRQ(ierr);

            if (sqrt(duNorm) <= tolerance) {
                break;
            };

            //Updates SUPG Parameter
            // for (int i = 0; i < numElem; i++){
            //     elements_[i] -> getParameterSUPG();
            // };
            
        };//Newton-Raphson

        // Compute and print drag and lift coefficients
        if (computeDragAndLift){
            dragAndLiftCoefficients(dragLift);
        };

        //if (printVorticity){
        //    for (int i = 0; i < numNodes; i++){
        //        nodes_[i] -> clearVorticity();
        //    };
        //    for (int jel = 0; jel < numElem; jel++){
        //        elements_[jel] -> computeVorticity();
        //    };
//
        //    for (int i = 0; i < numNodes; ++i){
        //        double vort = nodes_[i] -> getVorticity();
        //        double signal = vort / fabs(vort);
        //        int root;
        //        struct { 
        //            double val; 
        //            int   rank; 
        //        } in, out; 
//
        //        in.val = fabs(vort);
        //        in.rank = rank;
//
        //        MPI_Reduce(&in,&out,1,MPI_DOUBLE_INT,MPI_MAXLOC,root,PETSC_COMM_WORLD);
        //        MPI_Bcast(&out.val,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);
        //        MPI_Bcast(&out.rank,1,MPI_INT,0,PETSC_COMM_WORLD);
        //        MPI_Bcast(&signal,1,MPI_DOUBLE,out.rank,PETSC_COMM_WORLD);
//
        //        vort = out.val * signal;
//
        //        nodes_[i] -> setVorticity(vort); 
        //    }
//
//
        //};
//
        //Printing results
        printResults(iTimeStep);
        
    };

   
    return 0;
};



#endif
