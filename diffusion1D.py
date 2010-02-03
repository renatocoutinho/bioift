import fipy ## imports fipy, a finite volumes set of libraries

## sometimes there are modules within modules and you can import them
## by typing:
from fipy.meshes import grid1D ## this imports the grid1D module from fipy

## you can define variables of any kind in python.
## For example, the mesh size:
mesh_size = 50 ## this is an integer

## or the size of the domain
length = 1.0 ## this is a float (real) number

## or the size of the control volume:
deltax = length/float(mesh_size) ## note that I just forced an integer to become float

## you can create an instance of a one dimensional mesh by typing
mesh_example = grid1D.Grid1D(nx= mesh_size, dx = deltax)

## create a field with initial values
phi_o = 0
from fipy.variables.cellVariable import CellVariable
phi = CellVariable(name="solution variable", mesh=mesh_example, value=phi_o)

## Define boundary condition values
leftValue = 1.0
rightValue = 0

## Creation of boundary conditions
from fipy.boundaryConditions.fixedValue import FixedValue
BCs = (FixedValue(faces = mesh_example.getFacesRight(), value=rightValue),
       FixedValue(faces=mesh_example.getFacesLeft(),value=leftValue))


D = 1.0 ## diffusivity (m^2/s)

## Transient diffusion equation is defined next
from fipy.terms.explicitDiffusionTerm import ExplicitDiffusionTerm
from fipy.terms.transientTerm import TransientTerm
eqX = TransientTerm() == ExplicitDiffusionTerm(coeff = D)

timeStep = 0.09*deltax**2/(2*D) ## size of time step
steps = 9000 ## number of time-steps

## create a GUI
from fipy import viewers
viewer = viewers.make(vars = phi, limits={'datamin':0.0, 'datamax':1.0})

## iterate until you get tired
for step in range(steps):
    eqX.solve(var = phi, boundaryConditions = BCs, dt = timeStep) ## solving the equation
    viewer.plot() ## update the GUI


