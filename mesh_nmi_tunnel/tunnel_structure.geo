
// Outer Colinder

pCo0 = newp; Point(pCo0) = {0, 0, 0, mesh_outer};
pCo1 = newp; Point(pCo1) = {-Colinder_radius, 0, 0, mesh_outer};
pCo2 = newp; Point(pCo2) = {0, Colinder_radius, 0, mesh_outer};
pCo3 = newp; Point(pCo3) = {+Colinder_radius, 0, 0, mesh_outer};
pCo4 = newp; Point(pCo4) = {0, -Colinder_radius, 0, mesh_outer};

pCo5 = newp; Point(pCo5) = {0, 0, Colinder_height, mesh_outer};
pCo6 = newp; Point(pCo6) = {-Colinder_radius, 0, Colinder_height, mesh_outer};
pCo7 = newp; Point(pCo7) = {0, Colinder_radius, Colinder_height, mesh_outer};
pCo8 = newp; Point(pCo8) = {+Colinder_radius, 0, Colinder_height, mesh_outer};
pCo9 = newp; Point(pCo9) = {0, -Colinder_radius, Colinder_height, mesh_outer};

// Colinder bottom lines
lCo0 = newl; Circle(lCo0) = {pCo1, pCo0, pCo2};
lCo1 = newl; Circle(lCo1) = {pCo2, pCo0, pCo3};
lCo2 = newl; Circle(lCo2) = {pCo3, pCo0, pCo4};
lCo3 = newl; Circle(lCo3) = {pCo4, pCo0, pCo1};

// Colinder top lines
lCo4 = newl; Circle(lCo4) = {pCo6, pCo5, pCo7};
lCo5 = newl; Circle(lCo5) = {pCo7, pCo5, pCo8};
lCo6 = newl; Circle(lCo6) = {pCo8, pCo5, pCo9};
lCo7 = newl; Circle(lCo7) = {pCo9, pCo5, pCo6};

// Conlinder vertical lines
lCo8 = newl; Line(lCo8) = {pCo1, pCo6};
lCo9 = newl; Line(lCo9) = {pCo2, pCo7};
lCo10 = newl; Line(lCo10) = {pCo3, pCo8};
lCo11 = newl; Line(lCo11) = {pCo4, pCo9};

llCoUpper = newll; Line Loop(llCoUpper) = {lCo4,lCo5,lCo6,lCo7};
llCoLower = newll; Line Loop(llCoLower) = {lCo0,lCo1,lCo2,lCo3};

llCoVert1 = newll; Line Loop(llCoVert1) = {lCo0, lCo9, -lCo4, -lCo8};
llCoVert2 = newll; Line Loop(llCoVert2) = {lCo1, lCo10, -lCo5, -lCo9};
llCoVert3 = newll; Line Loop(llCoVert3) = {lCo2, lCo11, -lCo6, -lCo10};
llCoVert4 = newll; Line Loop(llCoVert4) = {lCo3, lCo8, -lCo7, -lCo11};
