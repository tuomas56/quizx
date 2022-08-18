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

use std::collections::{HashMap, HashSet};

use crate::graph::{VData, VType, EType, GraphLike};
use femtovg as fv;
use rand::Rng;

/// A struct representing one node of a graph.
#[derive(Debug, Clone)]
struct Node {
    /// Vertex space position
    pos: (f32, f32),
    /// Label above the node
    label_upper: Option<String>,
    /// Label beneath the node
    label_lower: Option<String>,
    /// Shape of the outline
    shape: fv::Path,
    /// Style of the interior
    fill: fv::Paint,
    /// Style of the border
    stroke: fv::Paint,
    /// Whether this node should be moved in a computed layout
    internal: bool,
    /// The size of the hitbox for this node
    radius: f32,
    /// The ID of this node in the underlying graph
    id: usize
}

/// A struct representing an edge of the graph
#[derive(Debug, Clone)]
struct Edge {
    /// The node index that this edge starts from
    start: usize,
    /// The node index that this edge ends at
    end: usize,
    /// The style of the line drawn between them
    stroke: fv::Paint
}

/// This struct handles the rendering and layout of
/// the graph and the button controls
struct Viewer {
    /// Canvas to draw on
    canvas: fv::Canvas<fv::renderer::OpenGl>,
    /// Width, height, dpi factor
    size: (u32, u32, f32),
    /// The color of the labels
    label_style: fv::Paint,
    /// The screen-space width of lines
    line_width: f32,
    /// The screen-space font size
    font_size: f32,
    /// The fill color for the highlight box
    highlight_fill: fv::Paint,
    /// The border color for the highlight box
    highlight_stroke: fv::Paint,
    /// The fill color for buttons 
    button_fill: fv::Paint,
    /// Border color for buttons
    button_stroke: fv::Paint,
    /// Fill color for buttons when hovered over
    hover_fill: fv::Paint,
    /// Fill color for buttons when pressed
    press_fill: fv::Paint,
    /// Whether a layout has been computed for this graph
    computed_layout: bool,
    /// All the nodes in the graph
    nodes: Vec<Node>,
    /// All the edges in the graph
    edges: Vec<Edge>
}

impl Viewer {
    /// Populate a viewer struct from a GraphLike object.
    fn populate(&mut self, graph: &impl GraphLike) {
        // Map all the graph IDs to node numbers:
        let mut map = HashMap::new();
        let mut fresh = self.nodes.len();
        
        for vertex in graph.vertices() {
            // Turn the vertex data of the graph vertex into a node
            let node = graph.vertex_data(vertex).to_node(vertex);
            self.nodes.push(node);
            map.insert(vertex, fresh);
            fresh += 1;
        }

        for (start, end, ty) in graph.edges() {
            // Get the drawing style of this edge type
            let stroke = ty.to_edge_style();
            // Lookup the node ids for these vertices and add an edge
            let start = map[&start];
            let end = map[&end];
            self.edges.push(Edge { start, end, stroke });
        }
    }

    /// Update the viewer when a graph is modified
    fn rebuild_graph(&mut self, graph: &impl GraphLike) {
        // Remove nodes that are no longer present
        for i in (0..self.nodes.len()).rev() {
            if !graph.contains_vertex(self.nodes[i].id) {
                self.nodes.swap_remove(i);
            }
        }

        // Make a map from old nodes to IDs
        let mut map = HashMap::new();
        let mut fresh = self.nodes.len();
        for (i, node) in self.nodes.iter().enumerate() {
            map.insert(node.id, i);
        }

        let mut internal = HashMap::new();
        for vertex in graph.vertices() {
            // Create a new node for every vertex
            let node = graph.vertex_data(vertex).to_node(vertex);
            if let Some(&id) = map.get(&vertex) {
                // If this was already present, then move it to its old location
                internal.insert(id, node.internal);
                let old_pos = self.nodes[id].pos;
                self.nodes[id] = node;
                self.nodes[id].pos = old_pos;
                // Set the position as fixed so that when we recompute layouts,
                // old nodes wont move.
                self.nodes[id].internal = false;
            } else {
                // Otherwise create a new node and put it in the map
                self.nodes.push(node);
                map.insert(vertex, fresh);
                fresh += 1;
            }
        }

        // Recreate all the edges from scratch
        self.edges.clear();
        for (start, end, ty) in graph.edges() {
            let stroke = ty.to_edge_style();
            let start = map[&start];
            let end = map[&end];
            self.edges.push(Edge { start, end, stroke });
        }

        if self.computed_layout && internal.len() < map.len() {
            // If we had a computed layout before and new nodes were added
            // recompute the layout, keeping all the old nodes fixed (since internal was set to false)
            self.compute_layout();
        }

        for (id, internal) in internal {
            // Reset the internal flag back to its original values
            // restoring any previously internal nodes to be unfixed again
            self.nodes[id].internal = internal;
        }
    }

    /// Clear the nodes, edges, and canvas state.
    fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.canvas.restore();
        self.computed_layout = false;
    }

    /// Center the current graph in screen space and scale it to fit.
    fn center_graph(&mut self) {
        // If the graph is empty, do nothing
        if self.nodes.len() == 0 {
            return
        }

        // Compute the bounding box of the graph
        let mut bbox = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for node in &self.nodes {
            bbox.0 = bbox.0.min(node.pos.0);
            bbox.1 = bbox.1.max(node.pos.0);
            bbox.2 = bbox.2.min(node.pos.1);
            bbox.3 = bbox.3.max(node.pos.1);
        }

        // Add some padding to the bounding box
        bbox.0 -= 1.0;
        bbox.1 += 1.0;
        bbox.2 -= 3.0;
        bbox.3 += 1.0;

        // Compute a scale factor to make it fit the display
        let width = self.size.0 as f32;
        let height = self.size.1 as f32;
        let sx = width / (bbox.1 - bbox.0);
        let sy = height / (bbox.3 - bbox.2);
        let s = sx.min(sy);
        // Compute the offset to make it centered
        let ax = width / 2.0 - s * (bbox.0 + bbox.1) / 2.0;
        let ay = height / 2.0 - s * (bbox.2 + bbox.3) / 2.0;
        // Push those parameters to the canvas
        self.canvas.restore();
        self.canvas.save();
        self.canvas.translate(ax, ay);
        self.canvas.scale(s, s);
    }

    /// Perform one step of the force-directed layout algorithm
    fn fdl_step(&mut self, c2: f32, nonadj: &HashSet<(usize, usize)>) -> f32 {
        let c1: f32 = 2.0;
        let c3: f32 = 1.0;
        let c4: f32 = 0.1;

        let mut forces = vec![(0.0f32, 0.0f32); self.nodes.len()];

        // Every edge contributes a logarithmic spring force
        for edge in &self.edges {
            if edge.start == edge.end { continue }
            let a = self.nodes[edge.start].pos;
            let b = self.nodes[edge.end].pos;
            // Calculate distance between endpoints
            let d = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
            if d == 0.0 { continue }
            // Find logarithmic spring force and add to force vector
            let strength = c1 * (d / c2).log10();
            let dir = ((b.0 - a.0) / d, (b.1 - a.1) / d);
            forces[edge.start].0 += dir.0 * strength;
            forces[edge.start].1 += dir.1 * strength;
            forces[edge.end].0 -= dir.0 * strength;
            forces[edge.end].1 -= dir.1 * strength;
        }

        // Every nonadjacent pair of vertices contributes a repulsion force
        for &(av, bv) in nonadj {
            let a = self.nodes[av].pos;
            let b = self.nodes[bv].pos;
            // Calculate distance between nodes and make inverse square force
            let d2 = (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
            let strength = c3 / d2;
            let d = d2.sqrt();
            if d == 0.0 { continue }
            let dir = ((b.0 - a.0) / d, (b.1 - a.1) / d);
            forces[av].0 -= dir.0 * strength;
            forces[av].1 -= dir.1 * strength;
            forces[bv].0 += dir.0 * strength;
            forces[bv].1 += dir.1 * strength;
        }

        // For every non-fixed node, move it according to the clamped force value
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if node.internal {
                node.pos.0 += c4 * forces[i].0.max(-0.5).min(0.5);
                node.pos.1 += c4 * forces[i].1.max(-0.5).min(0.5);
            }
        }

        // Compute the maximum potential displacement in this step and return it
        forces.into_iter()
            .map(|f| c4 * f.0.abs().max(f.1.abs()))
            .sum::<f32>()
    }

    /// Use force-directed layout to relayout this graph
    fn compute_layout(&mut self) {
        let mut rng = rand::thread_rng();

        // If its empty, do nothing
        if self.nodes.len() == 0 {
            return
        }

        // Find a bounding box of the nodes
        let mut bbox = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for node in &self.nodes {
            bbox.0 = bbox.0.min(node.pos.0);
            bbox.1 = bbox.1.max(node.pos.0);
            bbox.2 = bbox.2.min(node.pos.1);
            bbox.3 = bbox.3.max(node.pos.1);
        }

        // Use the bounding box to find the density of nodes
        // and thus the target edge length for the layout
        let c2 = 0.75 * ((bbox.1 - bbox.0) * (bbox.3 - bbox.2) / self.nodes.len() as f32).sqrt().max(1.0);

        // If there are overlapping nodes, give them a random displacement
        for i in 0..self.nodes.len() {
            for j in 0..i {
                let a = self.nodes[i].pos;
                let b = self.nodes[j].pos;
                let d = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
                if d == 0.0 {
                    self.nodes[i].pos.0 += rng.gen_range(-c2/2.0..c2/2.0);
                    self.nodes[j].pos.0 += rng.gen_range(-c2/2.0..c2/2.0);
                }
            }
        }

        // Construct a set of all nonadjacent pairs of 
        // nodes by starting with all pairs
        let mut nonadj = HashSet::new();
        for a in 0..self.nodes.len() {
            for b in 0..a {
                nonadj.insert((a, b));
            }
        }

        // And then remove all the adjacent pairs
        for edge in &self.edges {
            let a = edge.start.max(edge.end);
            let b = edge.start.min(edge.end); 
            nonadj.remove(&(a, b));
        }

        // Do steps of FDL until it converges
        let mut iters = 0;
        let mut energy = f32::INFINITY;
        while iters < 2000 {
            // Find the maximum potential displacement
            let new_energy = self.fdl_step(c2, &nonadj);
            // If the relative change in potential displacement is small
            // then we are done, otherwise go again.
            if (new_energy - energy).abs() / energy < 1e-6 {
                break
            } else {
                energy = new_energy;
                iters += 1
            }
        }

        self.computed_layout = true;
    }

    /// Move the viewport by an amount in screen-space
    fn pan(&mut self, dx: f32, dy: f32) {
        let s = self.canvas.transform().average_scale();
        self.canvas.translate(dx / s, dy / s);
    }

    // Check if this screen-space position intersects a node
    fn pick_node(&self, x: f32, y: f32) -> Option<usize> {
        // Transform to world space
        let (x, y) = self.canvas.transform().inversed().transform_point(x, y);

        // Use the radius of each node to find the first overlap
        for (i, node) in self.nodes.iter().enumerate() {
            let d2 = (x - node.pos.0).powi(2) + (y - node.pos.1).powi(2);
            if d2 <= node.radius * node.radius {
                return Some(i)
            }
        }

        None
    }

    // Add all nodes in the given screen-space rectangle to a target set.
    fn pick_nodes(&self, rect: Option<((f32, f32), (f32, f32))>, target: &mut HashSet<usize>) {
        if let Some(((x1, y1), (x2, y2))) = rect {
            // Transform the rectangle into world space
            let (x1, y1) = self.canvas.transform().inversed().transform_point(x1, y1);
            let (x2, y2) = self.canvas.transform().inversed().transform_point(x2, y2);
            // Find the equivalent bounding box
            let cx1 = x1.min(x2);
            let cx2 = x1.max(x2);
            let cy1 = y1.min(y2);
            let cy2 = y1.max(y2);
            
            // For every node in that box, add to the target
            for (i, node) in self.nodes.iter().enumerate() {
                if node.pos.0 >= cx1 && node.pos.0 <= cx2 && node.pos.1 >= cy1 && node.pos.1 <= cy2 {
                    target.insert(i);
                }
            }
        }
    }

    /// Move a set of nodes by a given screen-space amount
    fn move_nodes(&mut self, nodes: &HashSet<usize>, dx: f32, dy: f32) {
        let s = self.canvas.transform().average_scale();
        for &node in nodes {
            self.nodes[node].pos.0 += dx / s;
            self.nodes[node].pos.1 += dy / s;
        }
    }

    /// Zoom the view by a factor about a central screen-space point
    fn zoom(&mut self, factor: f32, cx: f32, cy: f32) {
        let before = self.canvas.transform().inversed().transform_point(cx, cy);
        self.canvas.scale(factor, factor);
        let after = self.canvas.transform().inversed().transform_point(cx, cy);
        self.canvas.translate(after.0 - before.0, after.1 - before.1);
    }

    /// Inform the viewer about changes in window size
    fn resize(&mut self, width: u32, height: u32) {
        self.size.0 = width;
        self.size.1 = height;
    }

    /// Render the graph and controls to the display
    fn draw(&mut self, 
        // The nodes that should be highlighted
        selected: &HashSet<usize>, 
        // The current highlighting box
        highlight: Option<((f32, f32), (f32, f32))>, 
        // Whether to render upper and lower labels
        labels: (bool, bool),
        // A list of buttons to render at the top
        buttons: &[&'static str],
        // The current location of the mouse
        mouse: Option<(f32, f32)>,
        // Whether the left mouse button is pressed
        mouse_down: bool
    ) {
        self.canvas.set_size(self.size.0, self.size.1, self.size.2);
        self.canvas.clear_rect(0, 0, self.size.0, self.size.1, fv::Color::white());

        // Compute world space line width and font size to achieve constant screen-space sizes
        let line_width = self.line_width / self.canvas.transform().average_scale();
        let font_size = 100.0 * self.font_size / self.canvas.transform().average_scale();

        for edge in &self.edges {
            // An edge is selected if either of the endpoints are
            let selected = selected.contains(&edge.start) || selected.contains(&edge.end);
            if edge.start != edge.end {
                // For non-self edges, draw a straight line between endpoints
                let (sx, sy) = self.nodes[edge.start].pos;
                let (ex, ey) = self.nodes[edge.end].pos;
                let mut path = fv::Path::new();
                path.move_to(sx, sy);
                path.line_to(ex, ey);
                self.canvas.stroke_path(&mut path, edge.stroke
                    .with_line_width(if selected {
                        // Selected edges are thicker
                        1.5 * line_width * edge.stroke.line_width()
                    } else {
                        1.0 * line_width * edge.stroke.line_width()   
                    }));
            } else {
                // For self-edges, draw a bezier curve that sticks out of the node
                let (x, y) = self.nodes[edge.start].pos;
                let mut path = fv::Path::new();
                path.move_to(x, y);
                path.bezier_to(x - 1.0, y - 1.0, x + 1.0, y - 1.0, x, y);
                self.canvas.stroke_path(&mut path, edge.stroke
                    .with_line_width(if selected {
                        1.5 * line_width * edge.stroke.line_width()
                    } else {
                        1.0 * line_width * edge.stroke.line_width()   
                    }));
            }
        }

        for (i, node) in self.nodes.iter_mut().enumerate() {
            self.canvas.save();
            // Translate to the nodes position and draw it
            self.canvas.translate(node.pos.0, node.pos.1);
            self.canvas.fill_path(&mut node.shape, node.fill);
            self.canvas.stroke_path(&mut node.shape, node.stroke
                .with_line_width(if selected.contains(&i) {
                    // Selected nodes have thicker borders
                    1.7 * line_width * node.stroke.line_width()
                } else {
                    1.0 * line_width * node.stroke.line_width()
                }));

            if labels.0 {
                // If we are drawing the top label, do it now
                if let Some(label) = node.label_upper.as_ref() {
                    self.canvas.save();
                    self.canvas.scale(0.01, 0.01);
                    self.canvas.fill_text(0.0, -15.0, label, self.label_style
                        .with_font_size(font_size)
                        .with_text_align(fv::Align::Center)
                        .with_text_baseline(fv::Baseline::Bottom)).unwrap();
                    self.canvas.restore();
                }
            }

            if labels.1 {
                // Same with the lower label
                if let Some(label) = node.label_lower.as_ref() {
                    self.canvas.save();
                    self.canvas.scale(0.01, 0.01);
                    self.canvas.fill_text(0.0, 15.0, label, self.label_style
                        .with_font_size(font_size)
                        .with_text_align(fv::Align::Center)
                        .with_text_baseline(fv::Baseline::Top)).unwrap();
                    self.canvas.restore();
                }
            }

            self.canvas.restore();
        }

        let world_to_canvas = self.canvas.transform().inversed();

        // If there is a highlight box
        if let Some(((x1, y1), (x2, y2))) = highlight {
            // Find the world space positions of the corners
            let (x1, y1) = world_to_canvas.transform_point(x1, y1);
            let (x2, y2) = world_to_canvas.transform_point(x2, y2);
            // Construct an equivalent bounding box
            let cx1 = x1.min(x2);
            let cx2 = x1.max(x2);
            let cy1 = y1.min(y2);
            let cy2 = y1.max(y2);
            // Draw the box on screen
            let mut highlight_path = fv::Path::new();
            highlight_path.rect(cx1, cy1, cx2 - cx1, cy2 - cy1);
            self.canvas.fill_path(&mut highlight_path, self.highlight_fill);
            self.canvas.stroke_path(&mut highlight_path, self.highlight_stroke
                .with_line_width(line_width));
        }

        // Find the mouse position in world space and the size of the buttons
        let mouse = mouse.map(|(x, y)| world_to_canvas.transform_point(x, y));
        let origin = world_to_canvas.transform_point(0.0, 0.0);
        let row_advance = world_to_canvas.transform_point(self.size.0 as f32, 15.0);
        let row_advance = (row_advance.0 - origin.0, row_advance.1 - origin.1);

        let mut button_path = fv::Path::new();
        button_path.rect(0.0, 0.0, row_advance.0 / 6.0, row_advance.1);

        self.canvas.save();
        // Go to the top left corner
        self.canvas.translate(origin.0, origin.1);

        // For each row of six buttons
        for (i, row) in buttons.chunks(6).enumerate() {
            // For every button in that row
            for (j, button) in row.iter().enumerate() {
                // Calculate whether the mouse is in that button
                let inside = if let Some((x, y)) = mouse {
                    y >= origin.1 + i as f32 * row_advance.1
                    && y < origin.1 + (i as f32 + 1.0) * row_advance.1
                    && x >= origin.0 + (j as f32) * row_advance.0 / 6.0
                    && x < origin.0 + (j as f32 + 1.0) * row_advance.0 / 6.0
                } else {
                    false
                };

                // Draw it on screen with appropriate fill color for mouse state
                self.canvas.fill_path(&mut button_path, if inside && mouse_down {
                    self.press_fill
                } else if inside {
                    self.hover_fill
                } else {
                    self.button_fill
                });

                self.canvas.stroke_path(&mut button_path, self.button_stroke
                    .with_line_width(line_width));

                // Draw the text in the middle (scaled for better resolution)
                self.canvas.scale(0.01, 0.01);
                self.canvas.fill_text(100.0 * row_advance.0 / 12.0, 0.0, button, self.label_style
                    .with_font_size(font_size)
                    .with_text_align(fv::Align::Center)
                    .with_text_baseline(fv::Baseline::Top)).unwrap();
                self.canvas.scale(100.0, 100.0);

                // Move along to the next button
                self.canvas.translate(row_advance.0 / 6.0, 0.0);
            }

            // Move to the next row
            self.canvas.translate(-row_advance.0, row_advance.1);
        }

        self.canvas.restore();

        self.canvas.flush();
    }
}

impl VType {
    /// Convert a vertex type to a style of node
    fn to_node_style(self) -> (fv::Path, fv::Paint, fv::Paint, bool, f32) {
        match self {
            VType::B => {
                // Boundaries are small black circles that shouldn't be moved (not internal)
                let mut path = fv::Path::new();
                path.circle(0.0, 0.0, 0.05);
                path.solidity(fv::Solidity::Solid);
                let paint = fv::Paint::default()
                    .with_color(fv::Color::black())
                    .with_line_width(1.0);
                (path, paint, paint, false, 0.05)
            },
            VType::Z => {
                // Z spiders are large green circles that can be moved
                let mut path = fv::Path::new();
                path.circle(0.0, 0.0, 0.1);
                path.solidity(fv::Solidity::Solid);
                let fill = fv::Paint::default()
                    .with_color(fv::Color::rgb(221, 255, 221));
                let stroke = fv::Paint::default()
                    .with_color(fv::Color::black())
                    .with_line_width(1.0);
                (path, fill, stroke, true, 0.1)
            },
            VType::X => {
                // Same for X but red
                let mut path = fv::Path::new();
                path.circle(0.0, 0.0, 0.1);
                path.solidity(fv::Solidity::Solid);
                let fill = fv::Paint::default()
                    .with_color(fv::Color::rgb(255, 136, 136));
                let stroke = fv::Paint::default()
                    .with_color(fv::Color::black())
                    .with_line_width(1.0);
                (path, fill, stroke, true, 0.1)
            },
            VType::H => {
                // H boxes are yellow squares that can be moved
                let mut path = fv::Path::new();
                path.rect(0.0, 0.0, 0.05, 0.05);
                path.solidity(fv::Solidity::Solid);
                let fill = fv::Paint::default()
                    .with_color(fv::Color::rgb(255, 255, 0));
                let stroke = fv::Paint::default()
                    .with_color(fv::Color::black())
                    .with_line_width(1.0);
                (path, fill, stroke, true, 0.05)
            }
        }
    }
}

impl EType {
    /// Construct a viewer edge line style from a graph edge type
    fn to_edge_style(self) -> fv::Paint {
        match self {
            EType::N => fv::Paint::default()
                .with_color(fv::Color::black())
                .with_line_width(1.0),
            EType::H => fv::Paint::default()
                .with_color(fv::Color::rgb(68, 136, 255))
                .with_line_width(1.0)
        }
    }
}

impl VData {
    // Construct a node for a given vertex from its VData
    fn to_node(&self, id: usize) -> Node {
        // Use the vertex type to create the style
        let (shape, fill, stroke, internal, radius) = self.ty.to_node_style();
        // Format the upper label as the phase
        let label_upper = if *self.phase.numer() == 0 {
            None
        } else if self.phase.is_integer() {
            match *self.phase.numer() {
                1 => Some("π".to_string()),
                -1 => Some("-π".to_string()),
                _ => unreachable!()
            }
        } else if *self.phase.numer() == 1 {
            Some(format!("π/{}", self.phase.denom()))
        } else if *self.phase.numer() == -1 {
            Some(format!("-π/{}", self.phase.denom()))
        } else {
            Some(format!("{}π/{}", self.phase.numer(), self.phase.denom()))
        };

        // The lower label is the node id
        let label_lower = Some(format!("{}", id));
        // The row and qubit position is converted to world space position
        let pos = (self.row as f32, self.qubit as f32);

        Node { 
            pos, label_upper, label_lower, shape, 
            fill, stroke, internal, radius, id 
        }
    }
}

/// This struct handles the interactions between
/// user input and the editor actions.
struct Editor<'a, G: GraphLike> {
    /// The underlying zx diagram that is being edited
    graph: &'a mut G,
    /// The original graph, if any, to restore to when needed
    original: Option<&'a G>,
    /// The viewer to render the graph
    viewer: Viewer,
    /// The window we are rendering in
    window: &'a glutin::window::Window,
    /// Whether the left mouse button is down
    mouse_down: bool,
    /// Whether the right mouse button is down
    right_down: bool,
    /// The current cursor position
    cursor_pos: Option<(f32, f32)>,
    /// The set of nodes currently selected
    selected: HashSet<usize>,
    /// The position of the start of the highlight box
    select_start: Option<(f32, f32)>,
    /// Whether to draw the upper and lower labels
    labels: (bool, bool),
    /// Whether a shift key is currently pressed
    shift_down: bool,
    /// Which rules can currently be applied to the diagram
    available_rules: Vec<&'static str>
}

impl<'a, G: GraphLike> Editor<'a, G> {
    /// Create a new editor from a graph and renderer backend
    fn new(
        graph: &'a mut G, 
        original: Option<&'a G>, 
        window: &'a glutin::window::Window, 
        renderer: fv::renderer::OpenGl
    ) -> Editor<'a, G> {
        let mut canvas = fv::Canvas::new(renderer).unwrap();

        // Find the window size and scale
        let dpi_factor = window.scale_factor();
        let size = window.inner_size();
    
        // Pick a generic font to use from the system fonts
        let query = font_loader::system_fonts::FontPropertyBuilder::new()
            .build();
        let font = font_loader::system_fonts::get(&query)
            .expect("Couldn't load fonts");
        let font_id = canvas.add_font_mem(&font.0)
            .expect("Couldn't load font");
    
        let label_style = fv::Paint::default()
            .with_color(fv::Color::black())
            .with_font(&[font_id]);
    
        let mut viewer = Viewer {
            canvas,
            size: (size.width as u32, size.height as u32, dpi_factor as f32),
            label_style,
            // Pick reasonable defaults for style parameters
            line_width: 1.2,
            font_size: 13.0,
            computed_layout: false,
            highlight_fill: fv::Paint::default()
                .with_color(fv::Color::rgba(84, 128, 255, 60)),
            highlight_stroke: fv::Paint::default()
                .with_color(fv::Color::rgba(84, 128, 255, 127)), 
            button_fill: fv::Paint::default()
                .with_color(fv::Color::rgba(100, 100, 100, 127)),
            button_stroke: fv::Paint::default()
                .with_color(fv::Color::rgba(50, 50, 50, 127)), 
            hover_fill: fv::Paint::default()
                .with_color(fv::Color::rgba(70, 70, 70, 127)),
            press_fill: fv::Paint::default()
                .with_color(fv::Color::rgba(50, 50, 50, 127)),
            nodes: Vec::new(),
            edges: Vec::new()
        };
    
        // Add nodes from the graph
        viewer.populate(graph);
        viewer.center_graph();

        // Start the editor in a state with no buttons down
        Editor {
            viewer, window,
            graph, original,
            mouse_down: false,
            right_down: false,
            cursor_pos: None,
            select_start: None,
            selected: HashSet::new(),
            labels: (true, false),
            shift_down: false,
            available_rules: Vec::new()
        }
    }

    /// Reset the graph back to the original 
    fn reset(&mut self) {
        self.viewer.clear();
        // If there is an original graph, reset the graph to that
        if let Some(graph) = self.original {
            *self.graph = graph.clone();
        }
        // Either way, rebuild the view with original positions
        self.viewer.populate(self.graph);
        self.viewer.center_graph();
    }

    /// Layout either the whole graph or the selected portion
    fn layout(&mut self) {
        if self.selected.len() > 0 {
            // If we have a selection, make everything outside that fixed
            // by changing internal to false
            let mut internal = Vec::new();
            for (i, node) in self.viewer.nodes.iter_mut().enumerate() {
                internal.push(node.internal);
                if !self.selected.contains(&i) {
                    node.internal = false;
                }
            }

            // Compute the layout, only moving the selected stuff
            self.viewer.compute_layout();

            // Unfix the rest again
            for (node, internal) in self.viewer.nodes.iter_mut().zip(internal) {
                node.internal = internal;
            }
        } else {
            // Otherwise, just layout everything
            self.viewer.compute_layout();
        }
    }

    fn compute_available_rules(&mut self) {
        // We can always apply all the simplification rules
        // as well as the layout, reset, center and toggle metarules
        self.available_rules = vec![
            "compute layout",
            "reset graph",
            "center graph",
            "toggle ids",
            "toggle phases",
            "clifford_simp",		
            "flow_simp",	
            "full_simp",
            "fuse_gadgets",	
            "gen_pivot_simp",
            "id_simp",
            "interior_clifford_simp",
            "local_comp_simp",
            "pivot_simp",
            "scalar_simp",
            "spider_simp",
        ];

        match self.selected.len() { 
            1 => {
                // If we have selected on thing, check to see if any of the 
                // vertex simplifications are available
                let mut iter = self.selected.iter().copied();
                let v0 = self.viewer.nodes[iter.next().unwrap()].id;

                if crate::basic_rules::check_color_change(self.graph, v0) {
                    self.available_rules.push("color_change");
                }

                if crate::basic_rules::check_local_comp(self.graph, v0) {
                    self.available_rules.push("local_comp");
                }

                if crate::basic_rules::check_remove_id(self.graph, v0) {
                    self.available_rules.push("remove_id");
                }

                if crate::basic_rules::check_remove_single(self.graph, v0) {
                    self.available_rules.push("remove_single");
                }

                if crate::basic_rules::check_pi_copy(self.graph, v0) {
                    self.available_rules.push("pi_copy");
                }
            },
            2 => {
                // If we have selected two things, check if any of the edge 
                // simplifications are possible to do
                let mut iter = self.selected.iter().copied();
                let v0 = self.viewer.nodes[iter.next().unwrap()].id;
                let v1 = self.viewer.nodes[iter.next().unwrap()].id;

                if crate::basic_rules::check_boundary_pivot(self.graph, v0, v1) {
                    self.available_rules.push("boundary_pivot");
                }

                if crate::basic_rules::check_gadget_fusion(self.graph, v0, v1) {
                    self.available_rules.push("gadget_fusion");
                }

                if crate::basic_rules::check_gen_pivot(self.graph, v0, v1) {
                    self.available_rules.push("gen_pivot");
                }

                if crate::basic_rules::check_pivot(self.graph, v0, v1) {
                    self.available_rules.push("pivot");
                }

                if crate::basic_rules::check_remove_pair(self.graph, v0, v1) {
                    self.available_rules.push("remove_pair");
                }

                if crate::basic_rules::check_spider_fusion(self.graph, v0, v1) {
                    self.available_rules.push("spider_fusion");
                }
            },
            // Any other number of selections, do nothing
            _ => ()
        }
    }

    /// Check if this mouse position corresponds to a rule button
    fn pick_button(&self, x: f32, y: f32) -> Option<&'static str> {
        let button_width = self.viewer.size.0 as f32 / 6.0;

        // Check every row of buttons
        for (i, row) in self.available_rules.as_slice().chunks(6).enumerate() {
            if y < 15.0 * i as f32 || y >= 15.0 * (i as f32 + 1.0) {
                continue
            }

            // For every button in that row, see if the mouse is inside
            for (j, button) in row.iter().enumerate() {
                if x >= button_width * j as f32 && x <= button_width * (j as f32 + 1.0) {
                    return Some(*button);
                }
            }
        }

        None
    }

    /// Apply a named rewrite rule or metarule to the editor
    fn apply_rule(&mut self, rule: &'static str) {
        // Simplification and metarules are applied directly
        match rule {
            "compute layout" => {
                self.layout();
                true
            },
            "reset graph" => {
                self.reset();
                true
            },
            "center graph" => {
                self.viewer.center_graph();
                true
            },
            "toggle ids" => {
                self.labels.1 = !self.labels.1;
                true
            },
            "toggle phases" => {
                self.labels.0 = !self.labels.0;
                true
            },
            "clifford_simp" => crate::simplify::clifford_simp(self.graph),		
            "flow_simp" => crate::simplify::flow_simp(self.graph),	
            "full_simp" => crate::simplify::full_simp(self.graph),
            "fuse_gadgets" => crate::simplify::fuse_gadgets(self.graph),	
            "gen_pivot_simp" => crate::simplify::gen_pivot_simp(self.graph),
            "id_simp" => crate::simplify::id_simp(self.graph),
            "interior_clifford_simp" => crate::simplify::interior_clifford_simp(self.graph),
            "local_comp_simp" => crate::simplify::local_comp_simp(self.graph),
            "pivot_simp" => crate::simplify::pivot_simp(self.graph),
            "scalar_simp" => crate::simplify::scalar_simp(self.graph),
            "spider_simp" => crate::simplify::spider_simp(self.graph),
            _ => false
        };

        match self.selected.len() { 
            1 => {
                // If we have one selected vertex, find it and apply the rule there
                let mut iter = self.selected.iter().copied();
                let v0 = self.viewer.nodes[iter.next().unwrap()].id;

                match rule {
                    "color_change" => crate::basic_rules::color_change(self.graph, v0),
                    "local_comp" => crate::basic_rules::local_comp(self.graph, v0),
                    "remove_id" => crate::basic_rules::remove_id(self.graph, v0),
                    "remove_single" => crate::basic_rules::remove_single(self.graph, v0),
                    "pi_copy" => crate::basic_rules::pi_copy(self.graph, v0),
                    _ => false
                };
            },
            2 => {
                // If we have two selected vertices, apply the rule there
                let mut iter = self.selected.iter().copied();
                let v0 = self.viewer.nodes[iter.next().unwrap()].id;
                let v1 = self.viewer.nodes[iter.next().unwrap()].id;

                match rule {
                    "boundary_pivot" => crate::basic_rules::boundary_pivot(self.graph, v0, v1),
                    "gadget_fusion" => crate::basic_rules::gadget_fusion(self.graph, v0, v1),
                    "gen_pivot" => crate::basic_rules::gen_pivot(self.graph, v0, v1),
                    "pivot" => crate::basic_rules::pivot(self.graph, v0, v1),
                    "remove_pair" => crate::basic_rules::remove_pair(self.graph, v0, v1),
                    "spider_fusion" => crate::basic_rules::spider_fusion(self.graph, v0, v1),
                    _ => false
                };
            },
            _ => ()
        }

        // Update the graph in the viewer
        self.viewer.rebuild_graph(self.graph);
    }

    /// Called when the left mouse button is pressed or released
    fn handle_left_mouse(&mut self) {
        if self.mouse_down {
            // If the mous is being pressed
            if let Some(pos) = self.cursor_pos {
                // Check if a button was pressed
                if let Some(rule) = self.pick_button(pos.0, pos.1) {
                    // If so, apply the rules, clear the selection and recompute rules
                    self.apply_rule(rule);
                    self.selected.clear();
                    self.compute_available_rules();
                } else if let Some(node) = self.viewer.pick_node(pos.0, pos.1) {
                    // Otherwise, if a node was clicked and that node is not currently selected
                    // make the selection just this node
                    if !self.selected.contains(&node) {
                        self.selected.clear();
                        self.selected.insert(node);
                        // Don't recompute available rules since this is a temporary selection
                    }
                } else {
                    // Otherwise we clicked on empty, so clear the selection (and recompute)
                    self.selected.clear();
                    self.compute_available_rules();
                }
            }
        } else {
            // If the mouse is released and only one thing was selected,
            // the this was a temporary selection so clear it
            if self.selected.len() == 1 {
                self.selected.clear();
                self.compute_available_rules();
            }
        }

        self.window.request_redraw();
    }

    /// Called when the right mouse button is pressed or released
    fn handle_right_mouse(&mut self) {
        if self.right_down {
            // If it was pressed, start the highlight box at this position
            self.select_start = self.cursor_pos;
        } else {
            // If it was released:
            if !self.shift_down {
                // If shift is held, we are adding to the current selection
                // and if not, this is a new selection so clear
                self.selected.clear();
            }
            // Pick the nodes lying in the selection box and add to the selected list
            self.viewer.pick_nodes(self.cursor_pos.zip(self.select_start), &mut self.selected);

            // Reset the start of the highlight box to none
            self.select_start = None;

            // Compute what rules are now available to apply
            self.compute_available_rules();
        }

        self.window.request_redraw();
    }

    /// Called when the cursor is moved to some position
    fn handle_cursor_move(&mut self, x: f32, y: f32) {
        if let Some(prev) = self.cursor_pos {
            // Compute the movement from previous
            let dx = x - prev.0;
            let dy = y - prev.1;

            if self.mouse_down {
                if self.selected.len() > 0 {
                    // If we have nodes selected, move them
                    self.viewer.move_nodes(&self.selected, dx, dy);
                } else {
                    // Otherwise move the viewport
                    self.viewer.pan(dx, dy);
                }
            }

            self.window.request_redraw();
        }
    }

    /// Called when the mouse is scrolled by a number of lines
    fn handle_scroll(&mut self, delta: f32) {
        if let Some(pos) = self.cursor_pos {
            // Zoom either in or about about the current mouse position
            let factor = 1.15f32.powf(delta);
            self.viewer.zoom(factor, pos.0, pos.1);
            self.window.request_redraw();
        }
    }

    /// Called when a standard (non-modifier) key is pressed
    fn handle_key_down(&mut self, keycode: glutin::event::VirtualKeyCode) {
        // Add keyboard shortcuts for some of the metarules
        match keycode {
            winit::event::VirtualKeyCode::L => {
                self.layout();
                self.window.request_redraw();
            },
            winit::event::VirtualKeyCode::R => {
                self.reset();
                self.window.request_redraw();
            },
            winit::event::VirtualKeyCode::C => {
                self.viewer.center_graph();
                self.window.request_redraw();
            },
            winit::event::VirtualKeyCode::I => {
                // Toggle drawing of vertex IDs
                self.labels.1 = !self.labels.1;
                self.window.request_redraw();  
            },
            winit::event::VirtualKeyCode::P => {
                // Toggle drawing of vertex phases
                self.labels.0 = !self.labels.0;
                self.window.request_redraw();  
            }
            _ => ()
        }
    }

    /// Handle an event coming from the event loop and return whether to quit
    /// and whether we need to swap the double buffers
    fn handle_event(&mut self, event: glutin::event::Event<()>) -> (bool, bool) {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    // If the close button was clicked, exit!
                    return (true, false);
                },
                winit::event::WindowEvent::Resized(_) => {
                    // If the window was resized, check the actual window we are displaying in
                    // and then inform the viewer
                    let size = self.window.inner_size();
                    self.viewer.resize(size.width, size.height);
                },
                winit::event::WindowEvent::MouseInput { 
                    button: winit::event::MouseButton::Left, 
                    state, .. 
                } => {
                    // Update the mouse state and call the handler
                    self.mouse_down = match state {
                        winit::event::ElementState::Pressed => true,
                        winit::event::ElementState::Released => false,
                    };

                    self.handle_left_mouse();
                },
                winit::event::WindowEvent::MouseInput { 
                    button: winit::event::MouseButton::Right, 
                    state, .. 
                } => {
                    // Same deal here and below too
                    self.right_down = match state {
                        winit::event::ElementState::Pressed => true,
                        winit::event::ElementState::Released => false,
                    };

                    self.handle_right_mouse();
                },
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    self.handle_cursor_move(position.x as f32, position.y as f32);
                    
                    self.cursor_pos = Some((position.x as f32, position.y as f32));
                },
                winit::event::WindowEvent::MouseWheel { delta, .. } => {
                    if let winit::event::MouseScrollDelta::LineDelta(_, delta) = delta {
                        self.handle_scroll(delta);
                    }
                },
                winit::event::WindowEvent::KeyboardInput { input, .. } => {
                    if input.state == winit::event::ElementState::Pressed {
                        if let Some(keycode) = input.virtual_keycode {
                            self.handle_key_down(keycode);
                        }
                    }
                },
                winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                    // When the modifier keys change, update the state of the shift key
                    self.shift_down = modifiers.shift();                    
                }
                _ => ()
            },
            winit::event::Event::RedrawRequested(_) => {
                // If a redraw was requested, render using theh editor state
                // and return saying we need to swap the double buffers
                self.viewer.draw(
                    &self.selected, 
                    self.select_start.zip(self.cursor_pos), 
                    self.labels, 
                    &self.available_rules, 
                    self.cursor_pos,
                    self.mouse_down
                );
                return (false, true);
            },
            _ => ()
        }

        (false, false)
    }
}

/// View a graph in the editor without modifying it.
/// 
/// Like [edit] but creates a copy of the graph before editing, 
/// so no changes will be made to the graph. See the docs for 
/// [edit] for keyboard shortcuts and controls.
pub fn view(graph: &impl GraphLike) {
    let mut graph_copy = graph.clone();
    run_editor(&mut graph_copy, Some(graph));
}

/// Edit a graph in the editor.
/// 
/// This function opens a window with a GUI graph editor
/// containing the given graph:
/// 
/// * By left clicking, you can pan the viewport and move nodes around,
/// as well as deselecting any currently selected nodes.
/// * By right clicking and dragging, you can select nodes and 
/// use the buttons at the top of the editor to apply rules to these nodes.
/// By holding shift you can add to the current selection with the same action.
/// * There are buttons at the top of the editor
/// to apply simplification rules to the whole graph.
/// * There are also buttons to layout the graph, center, or reset 
/// it to the original, and toggle whether vertex phases and IDs are displayed.
/// * There are also keyboard shortcuts for some of these tasks, such as
/// 'L' for layout, 'C' for center, 'R' for reset, 'I' to toggle IDs 
/// and 'P' to toggle phases.
/// * Note that only rules applicable to the currently selected 
/// vertices are shown.
/// * If nodes are selected, only those nodes will be layed out when the 
/// layout button is selected.
pub fn edit(graph: &mut impl GraphLike) {
    let mut graph_copy = graph.clone();
    run_editor(&mut graph_copy, Some(graph));
    *graph = graph_copy;
}

/// Run the editor with a graph to edit and the original to reset to
fn run_editor<G: GraphLike>(graph: &mut G, original: Option<&G>) {
    // Build an event loop and window for the editor
    let mut event_loop = winit::event_loop::EventLoop::new();
    let window_builder = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::<f32>::new(1000., 600.))
        .with_title("QuiZX Editor");
    let windowed_context = glutin::ContextBuilder::new()
        .build_windowed(window_builder, &event_loop)
        .unwrap();
    let windowed_context = unsafe { 
        windowed_context.make_current().unwrap() 
    };

    // Create a renderer and editor from the opengl context of this window
    let renderer = fv::renderer::OpenGl::new_from_glutin_context(&windowed_context)
        .expect("Cannot create renderer");
    let window = windowed_context.window();
    let mut editor = Editor::new(
        graph, original,
        window, renderer
    );
    editor.compute_available_rules();

    // Start running the event loop:
    use winit::platform::run_return::EventLoopExtRunReturn;
    event_loop.run_return(|event, _, flow| {
        // Tell the editor about this event:
        let (done, swap) = editor.handle_event(event);

        // If we are done, exit rather than polling for more events:
        if done {
            *flow = winit::event_loop::ControlFlow::Exit;
        } else {
            *flow = winit::event_loop::ControlFlow::Poll;
        }

        // If we just rendered something, swap the double buffer:
        if swap {
            windowed_context.swap_buffers().unwrap();
        }
    });

    // Due to a bug on some platforms where windows don't close properly 
    // after exiting, hide the window before we try and close it, 
    // so it doesn't show up even if it doesn't actually close.
    window.set_visible(false);
}
