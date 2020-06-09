
// Outer Cylinder

pCy0 = newp; Point(pCy0) = {0, 0, 0, mesh_outer};
pCy1 = newp; Point(pCy1) = {-cylinder_radius, 0, 0, mesh_outer};
pCy2 = newp; Point(pCy2) = {0, cylinder_radius, 0, mesh_outer};
pCy3 = newp; Point(pCy3) = {+cylinder_radius, 0, 0, mesh_outer};
pCy4 = newp; Point(pCy4) = {0, -cylinder_radius, 0, mesh_outer};

pCy5 = newp; Point(pCy5) = {0, 0, cylinder_height, mesh_outer};
pCy6 = newp; Point(pCy6) = {-cylinder_radius, 0, cylinder_height, mesh_outer};
pCy7 = newp; Point(pCy7) = {0, cylinder_radius, cylinder_height, mesh_outer};
pCy8 = newp; Point(pCy8) = {+cylinder_radius, 0, cylinder_height, mesh_outer};
pCy9 = newp; Point(pCy9) = {0, -cylinder_radius, cylinder_height, mesh_outer};

// Cylinder bottom lines
lCy0 = newl; Circle(lCy0) = {pCy1, pCy0, pCy2};
lCy1 = newl; Circle(lCy1) = {pCy2, pCy0, pCy3};
lCy2 = newl; Circle(lCy2) = {pCy3, pCy0, pCy4};
lCy3 = newl; Circle(lCy3) = {pCy4, pCy0, pCy1};

// cylinder top lines
lCy4 = newl; Circle(lCy4) = {pCy6, pCy5, pCy7};
lCy5 = newl; Circle(lCy5) = {pCy7, pCy5, pCy8};
lCy6 = newl; Circle(lCy6) = {pCy8, pCy5, pCy9};
lCy7 = newl; Circle(lCy7) = {pCy9, pCy5, pCy6};

// Cynlinder vertical lines
lCy8 = newl; Line(lCy8) = {pCy1, pCy6};
lCy9 = newl; Line(lCy9) = {pCy2, pCy7};
lCy10 = newl; Line(lCy10) = {pCy3, pCy8};
lCy11 = newl; Line(lCy11) = {pCy4, pCy9};

llCyUpper = newll; Line Loop(llCyUpper) = {lCy4,lCy5,lCy6,lCy7};
llCyLower = newll; Line Loop(llCyLower) = {lCy0,lCy1,lCy2,lCy3};

llCyVert1 = newll; Line Loop(llCyVert1) = {lCy0, lCy9, -lCy4, -lCy8};
llCyVert2 = newll; Line Loop(llCyVert2) = {lCy1, lCy10, -lCy5, -lCy9};
llCyVert3 = newll; Line Loop(llCyVert3) = {lCy2, lCy11, -lCy6, -lCy10};
llCyVert4 = newll; Line Loop(llCyVert4) = {lCy3, lCy8, -lCy7, -lCy11};

sCyVert1 = news; Surface(sCyVert1) =  {llCyVert1};
sCyVert2 = news; Surface(sCyVert2) =  {llCyVert2};
sCyVert3 = news; Surface(sCyVert3) =  {llCyVert3};
sCyVert4 = news; Surface(sCyVert4) =  {llCyVert4};

sCyUpper = news; Surface(sCyUpper) = {llCyUpper};



