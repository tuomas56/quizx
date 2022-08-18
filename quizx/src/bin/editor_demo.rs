// QuiZX - Rust library for quantum circuit rewriting and optimisation
//         using the ZX-calculus
// Copyright (C) 2021 - Aleks Kissinger
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashSet;

use num::Zero;
use quizx as zx;
use rand::prelude::IteratorRandom;
use zx::{graph::{BasisElem, GraphLike, EType, VType, VData}, hash_graph::Graph, tensor::ToTensor};

fn main() {
    // let mut circ = zx::circuit::Circuit::random()
    //     .qubits(20)
    //     .clifford_t(0.1)
    //     .depth(200)
    //     .build()
    //     .to_graph::<zx::hash_graph::Graph>();
    // circ.plug_inputs(&[zx::graph::BasisElem::X0; 20]);
    // circ.plug_outputs(&[zx::graph::BasisElem::X0; 20]);

    let circ = zx::circuit::Circuit::from_file("../circuits/small/adder_8.qasm").unwrap();
    let mut graph = circ.to_graph::<Graph>();
    zx::simplify::full_simp(&mut graph);
    let mut annealer = zx::annealer::Annealer::new(graph);
    annealer.scoref(|g| g.tcount());
    annealer.anneal();
}