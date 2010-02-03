
###
## Variables definitions
###
nx = 50
ny = nx
dx = 1.0/float(nx)
dy = dx
from fipy.meshes.grid2D import Grid2D
mesh = Grid2D(nx = nx, ny = ny, dx = dx, dy = dy)

from fipy.variables.cellVariable import CellVariable
phi = CellVariable(name="solution variable", mesh=mesh, value=0.4)
D = 1.0
a = 82.0
valueLeft = 0.0
valueRight = 0.
valueTop = 0.
valueBottom = 0.0
timeStepDuration = 20 * 0.9*dx**2/(2*D)
steps = 300


###
## Define Equation
###
from fipy.terms.explicitDiffusionTerm import ExplicitDiffusionTerm
from fipy.terms.implicitDiffusionTerm import ImplicitDiffusionTerm
from fipy.terms.transientTerm import TransientTerm
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
#from fipy.terms.sourceTerm import SourceTerm
#eqX = TransientTerm() == ExplicitDiffusionTerm(coeff = D) + _ExplicitSourceTerm(coeff=1.) - _ExplicitSourceTerm(coeff=phi)
eqI = TransientTerm() == ImplicitDiffusionTerm(coeff = D) + ImplicitSourceTerm(a * phi - a * phi * phi)
#eqCN = eqX + eqI

###
## Define Boundary Conditions
###
from fipy.boundaryConditions.fixedValue import FixedValue
BCs = (FixedValue(mesh.getFacesLeft(),valueLeft),
        FixedValue(faces = mesh.getFacesRight(), value=valueRight),
        FixedValue(mesh.getFacesTop(),valueTop),
        FixedValue(mesh.getFacesBottom(),valueBottom))


###
## Create a viewer
###
from fipy import Viewer
viewer = Viewer(vars = phi, datamin=0., datamax=1.0)


###
## Solve; Iterate
###
for step in range(steps):
    phi.updateOld()
    eqI.solve(var = phi, boundaryConditions = BCs, dt = timeStepDuration)
    print max(phi.value)
    viewer.plot()


