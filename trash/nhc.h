// Copyright 2013 Volodymyr Babin <vb27606@gmail.com>
//
// This is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// The code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You can find a copy of the GNU General Public License at
// http://www.gnu.org/licenses/.

#ifndef NHC_H
#define NHC_H

#include <cstddef>
#include "mt19937.h"

namespace kit { namespace nhc {

//
// one degree of freedom assumed ("massive thermostating")
//

size_t size(size_t M); // return size (in doubles) of a thermostat

void initialize(size_t M, double* thermo, const double& tau, mt19937&);

// returns velocity scale factor; Ek2 is twice kinetic energy in units of kT
double advance(size_t M, double* thermo, const double& tau,
               const double& Ek2kT, const double& dt);

// in units of kT
double invariant(size_t M, const double* thermo, const double& tau);

}} // namespace kit::nhc

#endif // NHC_H
