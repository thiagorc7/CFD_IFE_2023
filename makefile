MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
CSOURCES         = $(wildcard *.cpp)
FCOMPILER        = gfortran -O2
CXXFLAGS        += -w
CURRENT_DIR      = $(shell pwd)

# Diretorio onde a biblioteca Eigen esta localizada
EIGEN_DIR = /usr/include/eigen3

# Inclui diretorios da Eigen e headers adicionais
INCLUDES = -I$(EIGEN_DIR) -I${PETSC_DIR}/include -I$(CURRENT_DIR)

# Libs para PETSc e MPI
PETSC_LIBS = ${PETSC_KSP_LIB}
MPI_LIBS = -lmpi

# Adiciona as bibliotecas Eigen, PETSc, MPI e outras a linkagem
LIBS = -lboost_system -std=c++0x $(PETSC_LIBS) $(MPI_LIBS)
LIBS += -L$(EIGEN_DIR)

# Dependencias adicionais (headers)$(HEADERS:.h=.o)
HEADERS = propGeo.h funcaoForma.h funcaoFormaLagrange.h funcaoCoeficientesTriangular.h

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

f: $(GCH_FILES:.h=.gch) $(CSOURCES:.cpp=.o)
	@-${CLINKER} -o $@ $^ $(LIBS) 

debug: $(CSOURCES:.cpp=.o)
	@-${CLINKER} -o $@ $^ $(LIBS) -g
	@gdb debug

%.o: %.cpp
	@-${CLINKER} $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.h
	@-${CLINKER} $(CXXFLAGS) $(INCLUDES) -c $< -o $@


clear:
	@rm -f *.o *~ f *.vtu mirror* domain* *.mod *.dat ma26* tensao* esforc* saida omega.txt *.geo *.msh $(CURRENT_DIR)/src/*.gch

run1:
	@mpirun -np 1 ./f

run2:
	@mpirun -np 2 ./f

run3:
	@mpirun -np 3 ./f

run4:
	@mpirun -np 4 ./f

run5:
	@mpirun -np 5 ./f

run6:
	@mpirun -np 6 ./f

run7:
	@mpirun -np 7 ./f

run8:
	@mpirun -np 8 ./f -pc_factor_nonzeros_along_diagonal 1.e-8

run16:
	@mpirun -np 16 ./f
#	@mpirun -np 16 ./f -pc_type jacobi -ksp_type gmres -ksp_monitor_singular_value -ksp_gmres_restart 1000
