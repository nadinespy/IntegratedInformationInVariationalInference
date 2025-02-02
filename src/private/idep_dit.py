"""
Compute I_dep on binary bivariate data with the dit toolbox. This script is
prepared to be called from Matlab using a serialised data string as input, as
used in ELPh/DoubleRedundancyDiscreteIdep.m.

NOTE: The only way to return data back into Matlab is by stdout, so refrain
from printing anything else apart from the final result.

Pedro Mediano, Aug 2020
"""
import sys
import numpy as np
from collections import Counter

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

import dit
from dit.multivariate import coinformation

def discrete_idep_single_lattice(d, idep_atom='02'):
    """
    Given a 4-variate dit.Distribution d, calculate the PhiID of the mutual
    information between the two first and two last variables. Prints the
    resulting atoms.

    This version uses only a single 4-variate lattice and reads off all atoms
    at the same time.

    Inputs:
      d         -- object of type dit.Distribution
      idep_atom -- atom to calculate with the Idep double-unique information.
                   Can be one of '02', '03', '12', '13'. If sympy is not
                   installed, it must be '02' (default: '02')
    """

    if idep_atom == '02':
        phiid_pair = (0,2)
    elif idep_atom == '03':
        phiid_pair = (0,3)
    elif idep_atom == '12':
        phiid_pair = (0,2)
    elif idep_atom == '13':
        phiid_pair = (1,3)
    else:
        raise ValueError("Invalid value in idep_atom. Must be one of '02','03','12,'13'")

    if idep_atom != '02' and not SYMPY_AVAILABLE:
        raise ValueError("If sympy is not available, idep_atom can only be '02' (the default)")

    ## Calculate full I_dep lattice
    measure = {'I': lambda d: coinformation(d, [[0,1],[2,3]])}
    dd = dit.profiles.DependencyDecomposition(d, measures=measure)


    ## Read MI terms
    Ixa = dd.atoms[((0,2),(1,),(3,))]['I']
    Ixb = dd.atoms[((0,3),(1,),(2,))]['I']
    Iya = dd.atoms[((1,2),(0,),(3,))]['I']
    Iyb = dd.atoms[((1,3),(0,),(2,))]['I']

    Ixya = dd.atoms[((0,1,2),(3,))]['I']
    Ixyb = dd.atoms[((0,1,3),(2,))]['I']
    Ixab = dd.atoms[((0,2,3),(1,))]['I']
    Iyab = dd.atoms[((1,2,3),(0,))]['I']

    Ixyab = dd.atoms[((0,1,2,3),)]['I']


    ## Read single-target PID unique infos
    Ux_xyta = min(dd.delta(e, 'I') for e in dd.edges(((0,2),)) if (3,) in e[0])
    Ux_xytb = min(dd.delta(e, 'I') for e in dd.edges(((0,3),)) if (2,) in e[0])

    Ua_abtx = min(dd.delta(e, 'I') for e in dd.edges(((0,2),)) if (1,) in e[0])
    Ua_abty = min(dd.delta(e, 'I') for e in dd.edges(((1,2),)) if (0,) in e[0])

    # For the joint-target PIDs, pick the edges manually
    Ux_xytab = min([dd.atoms[((0,2,3),(0,1))]['I'],
                    dd.atoms[((0,2,3),(1,2,3),(0,1))]['I'] - dd.atoms[((1,2,3),(0,1))]['I'],
                    dd.atoms[((0,2,3),(1,))]['I'],
                    dd.atoms[((0,2,3),(1,2,3))]['I'] - dd.atoms[((1,2,3),(0,))]['I']])

    Ua_abtxy = min([dd.atoms[((0,1,2),(2,3))]['I'],
                    dd.atoms[((0,1,2),(0,1,3),(2,3))]['I'] - dd.atoms[((0,1,3),(2,3))]['I'],
                    dd.atoms[((0,1,2),(3,))]['I'],
                    dd.atoms[((0,1,2),(0,1,3))]['I'] - dd.atoms[((0,1,3),(2,))]['I']])


    ## Get PhiID unique info
    phiid_val = min(dd.delta(edge, 'I') for edge in dd.edges((phiid_pair,)))

    # Check atom is positive, and fix possible numerical errors from maxent projections
    if phiid_val < 0:
        phiid_val = 0


    if not SYMPY_AVAILABLE:
        assert(idep_atom == '02')

        # If sympy is not available, manually solve the equations for rtr
        xta_val = phiid_val
        rta_val = Ua_abtx - xta_val
        xtr_val = Ux_xyta - xta_val
        rtr_val = Ixa - rta_val - xtr_val - xta_val

    else:

        # If sympy is available, set up full system of equations
        rtr, rta, rtb, rts = sp.var('rtr rta rtb rts')
        xtr, xta, xtb, xts = sp.var('xtr xta xtb xts')
        ytr, yta, ytb, yts = sp.var('ytr yta ytb yts')
        str, sta, stb, sts = sp.var('str sta stb sts')

        if idep_atom == '02':
            phiid_atom = xta
        elif idep_atom == '03':
            phiid_atom = xtb
        elif idep_atom == '12':
            phiid_atom = yta
        elif idep_atom == '13':
            phiid_atom = ytb
        else:
            raise ValueError("Invalid value in idep_atom. Must be one of '02','03','12,'13'")

        eqs = [ \
            rtr + rta + xtr + xta - Ixa,
            rtr + rtb + ytr + ytb - Iyb,
            rtr + rtb + xtr + xtb - Ixb,
            rtr + rta + ytr + yta - Iya,
            rtr + rta + xtr + xta + ytr + yta + str + sta - Ixya,
            rtr + rtb + xtr + xtb + ytr + ytb + str + stb - Ixyb,
            rtr + xtr + rta + xta + rtb + xtb + rts + xts - Ixab,
            rtr + ytr + rta + yta + rtb + ytb + rts + yts - Iyab,
            rtr+xtr+ytr+str+ rta+xta+yta+sta+ rtb+xtb+ytb+stb+ rts+xts+yts+sts - Ixyab,
            xtr + xta + xtb + xts - Ux_xytab,
            xtr + xta - Ux_xyta,
            xtr + xtb - Ux_xytb,
            rta + xta + yta + sta - Ua_abtxy,
            rta + xta - Ua_abtx,
            rta + yta - Ua_abty,
            phiid_atom - phiid_val
            ]

        # Solve and return
        all_pid = [rtr, rta, rtb, rts, \
                   xtr, xta, xtb, xts, \
                   ytr, yta, ytb, yts, \
                   str, sta, stb, sts]

        sols = sp.solve(eqs)
        rtr_val = sols[rtr]

    return rtr_val


if __name__ == '__main__':

    ## Read arguments from stdin
    tau = int(sys.argv[1])
    bX  = sys.argv[2]
    L   = len(bX)//4

    ## Transform input data string into array
    # Assumes Fortran ordering, which is Matlab's way of flattening matrices
    c = np.reshape(bX[:-1].split('-'), (L,2), 'F')

    # Stack past and future in columns
    c = np.hstack((c[:-tau], c[tau:]))

    # Concatenate rows to get one str per timestep, e.g. '0010', '1100', ...
    v = np.array([''.join(r) for r in c])


    ## Enumerate possible outcomes, count, and build dit.Distribution
    outcomes = [format(i, '0%ib'%4) for i in range(16)]
    cntr = Counter(v)
    p = np.array([cntr[o] for o in outcomes])/(L-tau)

    dist = dit.Distribution(outcomes, p)


    ## Compute I_dep double-redundancy using dit
    redred = discrete_idep_single_lattice(dist)

    # Print result to stdout so it can be caught by Matlab
    print(redred)

